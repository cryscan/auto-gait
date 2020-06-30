#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <nlopt.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/EulerAngles>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

using std::vector;
using std::tuple;
using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix3Xd;
using Eigen::Map;
using Eigen::Block;
using Eigen::EulerAngles;
using Eigen::EulerSystem;
using autodiff::dual;
using autodiff::Vector3dual;
using autodiff::VectorXdual;
using autodiff::Matrix3dual;
using autodiff::Matrix3Xdual;
using autodiff::forward::gradient;
using autodiff::forward::jacobian;
using autodiff::forward::wrt;
using autodiff::forward::at;

using EulerAnglesZYXdual = EulerAngles<dual, EulerSystem<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_X>>;

constexpr double delta_seconds = 0.05;

constexpr int state_dims = 21;
constexpr int dynamics_dims = 12;

struct Plant {
    Vector3d gravity = Vector3d(0.0, 0.0, -9.8);
    double mass = 1.0;
    Matrix3d inertia = Matrix3d::Identity();

    Vector3d p = Vector3d(0.0, 0.0, -1.0);
    Vector3d b = Vector3d(1.0, 1.0, 0.5);

    double ground_height = 0;
    Vector3d ground_normal = Vector3d(0.0, 0.0, 1.0);
    double friction = 0.707;
};

struct StateControl {
    Matrix3Xdual matrix;
    Block<Matrix3Xdual, 3, 1, true> r, r_dt, theta, theta_dt, p, p_dt, f;

    const Plant& plant;

    template<typename T>
    explicit StateControl(const T& t, const Plant& plant);

    [[nodiscard]] Vector3dual angular_velocity() const;
    [[nodiscard]] auto dynamics() const;
    [[nodiscard]] auto state() const;

    [[nodiscard]] Vector3dual box() const;
    dual contact_product() const;

    [[nodiscard]] auto param() const;
    auto param();
};

template<typename T>
StateControl::StateControl(const T& t, const Plant& plant) :
        matrix(Map<const Matrix<double, 3, state_dims / 3>>(t.data())),
        r(matrix.col(0)),
        r_dt(matrix.col(1)),
        theta(matrix.col(2)),
        theta_dt(matrix.col(3)),
        p(matrix.col(4)),
        p_dt(matrix.col(5)),
        f(matrix.col(6)),
        plant{plant} {
}

Vector3dual StateControl::angular_velocity() const {
    auto cos_theta_y = cos(theta.y());
    auto sin_theta_y = sin(theta.y());
    auto cos_theta_z = cos(theta.z());
    auto sin_theta_z = sin(theta.z());
    Matrix3dual transform;
    transform
            <<
            cos_theta_y * cos_theta_z, -sin_theta_z, 0,
            cos_theta_y * sin_theta_z, cos_theta_z, 0,
            -sin_theta_y, 0, 1;
    return transform * theta_dt;
}

auto StateControl::dynamics() const {
    auto linear_acc = f + plant.mass * plant.gravity;
    auto angular_vel = angular_velocity();
    auto angular_acc = f.cross(r - p) - angular_vel.cross(plant.inertia * angular_vel);
    return (Matrix<dual, dynamics_dims, 1>() << r_dt, linear_acc, angular_acc, p_dt).finished();
}

auto StateControl::state() const {
    auto angular_vel = angular_velocity();
    return (Matrix<dual, dynamics_dims, 1>() << r, r_dt, angular_vel, p).finished();
}

auto StateControl::param() const {
    return Map<const Matrix<dual, state_dims, 1>>(matrix.data());
}

auto StateControl::param() {
    return Map<Matrix<dual, state_dims, 1>>(matrix.data());
}

Vector3dual StateControl::box() const {
    EulerAnglesZYXdual rotation(theta.z(), theta.y(), theta.x());
    return (rotation * (p - r) - plant.p).cwiseAbs() - plant.b;
}

dual StateControl::contact_product() const {
    return f.norm() * p_dt.norm();
}

class Collocation {
    const Plant& plant;

    static auto func(const StateControl& current, const StateControl& next) {
        auto delta = next.state() - current.state();
        auto sum = next.dynamics() + current.dynamics();
        return delta - 0.5 * delta_seconds * sum;
    }

public:
    static constexpr unsigned output_dims = dynamics_dims;
    static constexpr unsigned frames = 2;

    explicit Collocation(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(unsigned index, const Input& input, Output& output, Grad& grad) {
        StateControl current(input.col(index), plant);
        StateControl next(input.col(index + 1), plant);

        VectorXdual output_dual;
        grad << jacobian(func, wrt(current.param(), next.param()), at(current, next), output_dual);
        output << output_dual.cast<double>();
    }

    template<typename Input, typename Output>
    void operator()(unsigned index, const Input& input, Output& output) {
        StateControl current(input.col(index), plant);
        StateControl next(input.col(index + 1), plant);
        auto output_dual = func(current, next);
        output << output_dual.cast<double>();
    }
};

class JointLimit {
    const Plant& plant;

public:
    static constexpr unsigned output_dims = 3;
    static constexpr unsigned frames = 1;

    explicit JointLimit(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(unsigned index, const Input& input, Output& output, Grad& grad) {
        StateControl state_control(input.col(index), plant);
        auto func = [&state_control] { return state_control.box(); };

        VectorXdual output_dual;
        grad << jacobian(func, wrt(state_control.param()), at(), output_dual);
        output << output_dual.cast<double>();
    }

    template<typename Input, typename Output>
    void operator()(unsigned index, const Input& input, Output& output) {
        StateControl state_control(input.col(index), plant);
        auto output_dual = state_control.box();
        output << output_dual.cast<double>();
    }
};

class Ground {
    const Plant& plant;

public:
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned frames = 1;

    explicit Ground(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(unsigned index, const Input& input, Output& output, Grad& grad) {
        StateControl state_control(input.col(index), plant);
        auto func = [&state_control, this] { return plant.ground_height - state_control.p.z(); };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void operator()(unsigned index, const Input& input, Output& output) {
        StateControl state_control(input.col(index), plant);
        dual output_dual = plant.ground_height - state_control.p.z();
        output << static_cast<double>(output_dual);
    }
};

class Friction {
    const Plant& plant;

public:
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned frames = 1;

    explicit Friction(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(unsigned index, const Input& input, Output& output, Grad& grad) {
        StateControl state_control(input.col(index), plant);
        auto func = [&state_control, this] {
            auto cos = state_control.f.dot(plant.ground_normal) / state_control.f.norm();
            return plant.friction - cos;
        };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void operator()(unsigned index, const Input& input, Output& output) {
        StateControl state_control(input.col(index), plant);
        auto cos = state_control.f.dot(plant.ground_normal) / state_control.f.norm();
        dual output_dual = plant.friction - cos;
        output << static_cast<double>(output_dual);
    }
};

class Contact {
    const Plant& plant;

public:
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned frames = 1;

    explicit Contact(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(unsigned index, const Input& input, Output& output, Grad& grad) {
        StateControl state_control(input.col(index), plant);
        auto func = [&state_control] { return state_control.contact_product(); };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void operator()(unsigned index, const Input& input, Output& output) {
        StateControl state_control(input.col(index), plant);
        dual output_dual = state_control.contact_product();
        output << static_cast<double>(output_dual);
    }
};

template<typename Func>
void
wrapper(unsigned output_dims, double* output, unsigned input_dims, const double* input, double* grad, void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    constexpr auto func_output_dims = Func::output_dims;
    auto horizon = output_dims / func_output_dims;

    Map<const MatrixXd> input_matrix(input, state_dims, input_dims / state_dims);
    Map<MatrixXd> output_matrix(output, func_output_dims, horizon);
    Map<MatrixXd> grad_matrix(grad, output_dims, input_dims);

    for (unsigned i = 0; i < horizon; ++i) {
        auto output_block = output_matrix.col(i);
        Func func(plant);

        if (grad != nullptr) {
            constexpr auto func_input_dims = Func::frames * state_dims;
            auto start_row = i * output_dims;
            auto start_col = i * state_dims;
            auto block = grad_matrix.block<func_output_dims, func_input_dims>(start_row, start_col);
            func(i, input_matrix, output_block, block);
        } else func(i, input_matrix, output_block);
    }
}

int main() {
    Plant plant;

    unsigned horizon = 40;
    std::vector<double> tol(state_dims * horizon, 0.1);
    nlopt::opt opt(nlopt::algorithm::LD_SLSQP, state_dims * horizon);
    opt.add_equality_mconstraint(wrapper<Collocation>, &plant, tol);
    opt.add_inequality_mconstraint(wrapper<JointLimit>, &plant, tol);
    opt.add_inequality_mconstraint(wrapper<Ground>, &plant, tol);
    opt.add_inequality_mconstraint(wrapper<Friction>, &plant, tol);
    opt.add_equality_mconstraint(wrapper<Contact>, &plant, tol);

    return 0;
}