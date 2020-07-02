#include <iostream>
#include <fstream>
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

using std::cout;
using std::endl;
using std::vector;
using std::tuple;
using Eigen::IOFormat;
using Eigen::MatrixBase;
using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix3Xd;
using Eigen::Ref;
using Eigen::Map;
using Eigen::Block;
using Eigen::RowMajor;
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

constexpr unsigned state_dims = 21;
constexpr unsigned dynamics_dims = 12;
constexpr unsigned boundary_dims = 6;

struct Plant {
    Vector3d gravity = Vector3d(0.0, 0.0, -9.8);
    double mass = 1.0;
    Matrix3d inertia = Matrix3d::Identity();

    Vector3d p = Vector3d(0.0, 0.0, -1.0);
    Vector3d b = Vector3d(0.1, 0.1, 0.1);

    double ground_height = 0;
    Vector3d ground_normal = Vector3d(0.0, 0.0, 1.0);
    double friction = 0.707;

    Vector3d init_pos = Vector3d(0.0, 0.0, 1.0);
    Vector3d init_vel = Vector3d::Zero();
    Vector3d init_dir = Vector3d::Zero();
    Vector3d final_pos = Vector3d(1.0, 0.0, 1.0);
    Vector3d final_vel = Vector3d::Zero();
    Vector3d final_dir = Vector3d::Zero();
};

struct StateControl {
    Matrix3Xdual matrix;
    Block<Matrix3Xdual, 3, 1, true> r, r_dt, theta, theta_dt, p, p_dt, f;

    const Plant& plant;

    explicit StateControl(const Ref<const VectorXd>& t, const Plant& plant);

    [[nodiscard]] Vector3dual angular_velocity() const;
    [[nodiscard]] auto dynamics() const;
    [[nodiscard]] auto state() const;

    [[nodiscard]] Vector3dual box() const;
    [[nodiscard]] dual ground() const;
    [[nodiscard]] dual friction() const;
    [[nodiscard]] dual contact() const;
    [[nodiscard]] dual energy() const;

    [[nodiscard]] VectorXdual init_boundary() const;
    [[nodiscard]] VectorXdual final_boundary() const;

    [[nodiscard]] auto param() const;
    auto param();
};

StateControl::StateControl(const Ref<const VectorXd>& t, const Plant& plant) :
        matrix(Map<const Matrix3Xd>(t.data(), 3, state_dims / 3)),
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
    auto linear_acc = f / plant.mass + plant.gravity;
    auto angular_vel = angular_velocity();
    auto angular_acc = plant.inertia.inverse() * (f.cross(r - p) - angular_vel.cross(plant.inertia * angular_vel));
    return (VectorXdual(dynamics_dims) << r_dt, linear_acc, angular_acc, p_dt).finished();
}

auto StateControl::state() const {
    auto angular_vel = angular_velocity();
    return (VectorXdual(dynamics_dims) << r, r_dt, angular_vel, p).finished();
}

auto StateControl::param() const {
    return Map<const VectorXdual>(matrix.data(), state_dims);
}

auto StateControl::param() {
    return Map<VectorXdual>(matrix.data(), state_dims);
}

Vector3dual StateControl::box() const {
    EulerAnglesZYXdual rotation(theta.z(), theta.y(), theta.x());
    auto distance = (rotation * (p - r) - plant.p).cwiseAbs2();
    return distance - plant.b.cwiseAbs2();
}

dual StateControl::ground() const {
    dual distance = p.z() - plant.ground_height;
    return -distance;
}

dual StateControl::friction() const {
    return -f.dot(plant.ground_normal);
}

dual StateControl::contact() const {
    return f.squaredNorm() * p_dt.squaredNorm();
}

dual StateControl::energy() const {
    return f.squaredNorm();
}

VectorXdual StateControl::init_boundary() const {
    auto delta_pos = r - plant.init_pos;
    auto delta_vel = r_dt - plant.init_vel;
    // auto delta_dir = theta - plant.init_dir;
    return (VectorXdual(boundary_dims) << delta_pos, delta_vel).finished();
}

VectorXdual StateControl::final_boundary() const {
    auto delta_pos = r - plant.final_pos;
    auto delta_vel = r_dt - plant.final_vel;
    // auto delta_dir = theta - plant.final_dir;
    return (VectorXdual(boundary_dims) << delta_pos, delta_vel).finished();
}

template<typename Derived>
class FuncBase {
protected:
    const Plant& plant;

public:
    static constexpr unsigned output_dims = Derived::output_dims;
    static constexpr unsigned num_states = Derived::num_states;

    explicit FuncBase(const Plant& plant) : plant{plant} {}

    template<typename Input, typename Output, typename Grad>
    void operator()(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        static_cast<Derived*>(this)->impl(input, output, grad);
    }

    template<typename Input, typename Output>
    void operator()(const MatrixBase<Input>& input, MatrixBase<Output>& output) {
        static_cast<Derived*>(this)->impl(input, output);
    }
};

class Collocation : public FuncBase<Collocation> {
    static VectorXdual func(const StateControl& current, const StateControl& next) {
        VectorXdual delta = next.state() - current.state();
        VectorXdual sum = next.dynamics() + current.dynamics();
        return delta - 0.5 * delta_seconds * sum;
    }

public:
    static constexpr unsigned output_dims = dynamics_dims;
    static constexpr unsigned num_states = 2;

    explicit Collocation(const Plant& plant) : FuncBase<Collocation>(plant) {}

    template<typename Input, typename Output, typename Grad>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        StateControl current(input.col(0), plant);
        StateControl next(input.col(1), plant);

        VectorXdual output_dual;
        grad << jacobian(func, wrt(current.param(), next.param()), at(current, next), output_dual).transpose();
        output << output_dual.cast<double>();
    }

    template<typename Input, typename Output>
    void impl(const Input& input, Output& output) {
        StateControl current(input.col(0), plant);
        StateControl next(input.col(1), plant);
        auto output_dual = func(current, next);
        output << output_dual.cast<double>();
    }
};

struct JointLimit : FuncBase<JointLimit> {
    static constexpr unsigned output_dims = 3;
    static constexpr unsigned num_states = 1;

    explicit JointLimit(const Plant& plant) : FuncBase<JointLimit>(plant) {}

    template<typename Input, typename Output, typename Grad>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        StateControl state_control(input, plant);
        auto func = [&] { return state_control.box(); };

        VectorXdual output_dual;
        grad << jacobian(func, wrt(state_control.param()), at(), output_dual).transpose();
        output << output_dual.cast<double>();
    }

    template<typename Input, typename Output>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output) {
        StateControl state_control(input, plant);
        auto output_dual = state_control.box();
        output << output_dual.cast<double>();
    }
};

struct Ground : public FuncBase<Ground> {
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned num_states = 1;

    explicit Ground(const Plant& plant) : FuncBase<Ground>(plant) {}

    template<typename Input, typename Output, typename Grad>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        StateControl state_control(input, plant);
        auto func = [&] { return state_control.ground(); };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output) {
        StateControl state_control(input, plant);
        output << static_cast<double>(state_control.ground());
    }
};

struct Friction : public FuncBase<Friction> {
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned num_states = 1;

    explicit Friction(const Plant& plant) : FuncBase<Friction>(plant) {}

    template<typename Input, typename Output, typename Grad>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        StateControl state_control(input, plant);
        auto func = [&] { return state_control.friction(); };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output) {
        StateControl state_control(input, plant);
        output << static_cast<double>(state_control.friction());
    }
};

struct Contact : public FuncBase<Contact> {
    static constexpr unsigned output_dims = 1;
    static constexpr unsigned num_states = 1;

    explicit Contact(const Plant& plant) : FuncBase<Contact>(plant) {}

    template<typename Input, typename Output, typename Grad>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output, MatrixBase<Grad>& grad) {
        StateControl state_control(input, plant);
        auto func = [&] { return state_control.contact(); };

        dual output_dual;
        grad << gradient(func, wrt(state_control.param()), at(), output_dual);
        output << static_cast<double>(output_dual);
    }

    template<typename Input, typename Output>
    void impl(const MatrixBase<Input>& input, MatrixBase<Output>& output) {
        StateControl state_control(input, plant);
        output << static_cast<double>(state_control.contact());
    }
};

template<typename Derived>
void
wrapper(unsigned output_dims, double* output, unsigned input_dims, const double* input, double* grad, void* data) {
    using Func = FuncBase<Derived>;
    auto& plant = *reinterpret_cast<Plant*>(data);

    constexpr auto func_input_dims = Func::num_states * state_dims;
    constexpr auto func_output_dims = Func::output_dims;
    auto horizon = input_dims / state_dims;
    auto func_horizon = output_dims / func_output_dims;

    Map<const MatrixXd> input_matrix(input, state_dims, horizon);
    Map<MatrixXd> output_matrix(output, func_output_dims, func_horizon);
    Map<MatrixXd> grad_matrix(grad, input_dims, output_dims);

    for (unsigned i = 0; i < func_horizon; ++i) {
        auto input_block = input_matrix.middleCols<Func::num_states>(i);
        auto output_block = output_matrix.col(i);
        Func func(plant);

        if (grad != nullptr) {
            auto start_row = i * state_dims;
            auto start_col = i * func_output_dims;
            auto grad_block = grad_matrix.block<func_input_dims, func_output_dims>(start_row, start_col);
            func(input_block, output_block, grad_block);
        } else func(input_block, output_block);
    }
}

/*
double boundary(unsigned input_dims, const double* input, double* grad, void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    auto horizon = input_dims / state_dims;
    Map<const MatrixXd> input_matrix(input, state_dims, horizon);
    Map<MatrixXd> grad_matrix(grad, state_dims, horizon);

    double output = 0;
    {
        StateControl state_control(input_matrix.col(0), plant);
        if (grad != nullptr) {
            auto func = [&] { return state_control.init_distance(); };
            dual output_dual;
            grad_matrix.col(0) << gradient(func, wrt(state_control.param()), at(), output_dual);
            output += static_cast<double>(output_dual);
        } else output += static_cast<double>(state_control.init_distance());
    }
    {
        StateControl state_control(input_matrix.col(horizon - 1), plant);
        if (grad != nullptr) {
            auto func = [&] { return state_control.final_distance(); };
            dual output_dual;
            grad_matrix.col(horizon - 1) << gradient(func, wrt(state_control.param()), at(), output_dual);
            output += static_cast<double>(output_dual);
        } else output += static_cast<double>(state_control.final_distance());
    }
    return output;
}
 */

void init_boundary(unsigned output_dims,
                   double* output,
                   unsigned input_dims,
                   const double* input,
                   double* grad,
                   void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    Map<const MatrixXd> input_matrix(input, state_dims, input_dims / state_dims);
    Map<VectorXd> output_matrix(output, output_dims);
    Map<MatrixXd> grad_matrix(grad, input_dims, output_dims);

    VectorXdual output_dual;
    StateControl state_control(input_matrix.col(0), plant);
    if (grad != nullptr) {
        auto func = [&] { return state_control.init_boundary(); };
        grad_matrix.topRows<state_dims>() << jacobian(func, wrt(state_control.param()), at(), output_dual).transpose();
    } else output_dual = state_control.init_boundary();
    output_matrix << output_dual.cast<double>();
}

void final_boundary(unsigned output_dims,
                    double* output,
                    unsigned input_dims,
                    const double* input,
                    double* grad,
                    void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    auto horizon = input_dims / state_dims;
    Map<const MatrixXd> input_matrix(input, state_dims, horizon);
    Map<VectorXd> output_matrix(output, output_dims);
    Map<MatrixXd> grad_matrix(grad, input_dims, output_dims);

    VectorXdual output_dual;
    StateControl state_control(input_matrix.col(horizon - 1), plant);
    if (grad != nullptr) {
        auto func = [&] { return state_control.final_boundary(); };
        grad_matrix.bottomRows<state_dims>()
                << jacobian(func, wrt(state_control.param()), at(), output_dual).transpose();
    } else output_dual = state_control.final_boundary();
    output_matrix << output_dual.cast<double>();
}

double energy(unsigned input_dims, const double* input, double* grad, void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    auto horizon = input_dims / state_dims;
    Map<const MatrixXd> input_matrix(input, state_dims, horizon);
    Map<MatrixXd> grad_matrix(grad, state_dims, horizon);

    double output = 0;
    for (unsigned i = 0; i < horizon - 1; ++i) {
        StateControl current(input_matrix.col(i), plant);
        StateControl next(input_matrix.col(i + 1), plant);
        auto func = [&] { return (current.energy() + next.energy()) * delta_seconds / 2; };

        if (grad != nullptr) {
            dual output_current, output_next;
            grad_matrix.col(i) += gradient(func, wrt(current.param()), at(), output_current);
            grad_matrix.col(i + 1) += gradient(func, wrt(next.param()), at(), output_next);
            dual output_dual = output_current + output_next;
            output += static_cast<double>(output_dual);
        } else {
            dual output_dual = func();
            output += static_cast<double>(output_dual);
        }
    }
    return output;
}

double rand_disturb(double min, double max) {
    double f = (double) rand() / RAND_MAX;
    return min + f * (max - min);
}

vector<double> init_guess(const Plant& plant, unsigned horizon) {
    vector<double> input(state_dims * horizon);
    Map<MatrixXd> matrix(input.data(), state_dims, horizon);
    for (unsigned i = 0; i < horizon; ++i) {
        VectorXd init(state_dims), final(state_dims);
        // Vector3d vel = (plant.final_pos - plant.init_pos) / (delta_seconds * horizon);
        Vector3d vel = Vector3d::Zero();

        Vector3d init_p_pos = plant.init_pos + plant.p;
        Vector3d final_p_pos = plant.final_pos + plant.p;
        // Vector3d p_vel = (final_p_pos - init_p_pos) / (delta_seconds * horizon);
        Vector3d p_vel = Vector3d::Zero();

        Vector3d force = -plant.mass * plant.gravity;
        // Vector3d force = Vector3d::Zero();

        init << plant.init_pos, vel, plant.init_dir, Vector3d::Zero(), init_p_pos, p_vel, force;
        final << plant.final_pos, vel, plant.final_dir, Vector3d::Zero(), final_p_pos, p_vel, force;
        matrix.col(i) << init + i * (final - init) / (horizon - 1);
    }
    // for (auto& x : input) x += rand_disturb(-0.001, 0.001);
    return input;
}

template<typename Derived>
constexpr unsigned dims(unsigned horizon) {
    using Func = FuncBase<Derived>;
    return Func::output_dims * (horizon - (Func::num_states - 1));
}

template<typename Derived>
auto tol(unsigned horizon, double value = 0) {
    return vector<double>(dims<FuncBase<Derived>>(horizon), value);
}

const IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

int main() {
    Plant plant;

    unsigned horizon = 40;
    nlopt::opt local_opt(nlopt::algorithm::LD_MMA, state_dims * horizon);
    local_opt.set_ftol_abs(1e-4);
    local_opt.set_ftol_rel(1e-2);
    local_opt.set_xtol_abs(1e-4);
    local_opt.set_xtol_rel(1e-2);

    nlopt::opt opt(nlopt::algorithm::LD_SLSQP, state_dims * horizon);
    opt.set_maxeval(100);
    // opt.set_local_optimizer(local_opt);
    opt.set_min_objective(energy, &plant);
    opt.add_equality_mconstraint(wrapper<Collocation>, &plant, tol<Collocation>(horizon, 0.01));
    opt.add_inequality_mconstraint(wrapper<JointLimit>, &plant, tol<JointLimit>(horizon, 0.01));
    // opt.add_inequality_mconstraint(wrapper<Ground>, &plant, tol<Ground>(horizon, 0.01));
    // opt.add_inequality_mconstraint(wrapper<Friction>, &plant, tol<Friction>(horizon));
    // opt.add_equality_mconstraint(wrapper<Contact>, &plant, tol<Contact>(horizon));
    opt.add_equality_mconstraint(init_boundary, &plant, vector<double>(boundary_dims, 0.01));
    opt.add_equality_mconstraint(final_boundary, &plant, vector<double>(boundary_dims, 0.01));

    auto x = init_guess(plant, horizon);
    Map<MatrixXd> matrix(x.data(), state_dims, horizon);
    cout << matrix << endl << endl;
    double f;

    try {
        opt.optimize(x, f);
    } catch (std::exception& e) {
        cout << e.what() << endl;
    }

    cout << matrix << endl;
    cout << f << endl;

    std::fstream fs("output.txt", std::ios::out);
    for (int i = 0; i < matrix.cols(); ++i) {
        fs << i;
        if (i != matrix.cols() - 1) fs << ", ";
        else fs << endl;
    }
    fs << matrix.format(csv_format) << endl;

    return 0;
}