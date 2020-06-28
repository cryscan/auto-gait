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
#include <unsupported/Eigen/CXX11/Tensor>
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
using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::TensorRef;
using Eigen::sin;
using Eigen::cos;
using autodiff::dual;
using autodiff::Vector3dual;
using autodiff::VectorXdual;
using autodiff::Matrix3dual;
using autodiff::Matrix3Xdual;
using autodiff::forward::jacobian;
using autodiff::forward::wrt;
using autodiff::forward::at;

constexpr double delta_seconds = 0.05;

constexpr int state_dims = 21;
constexpr int dynamics_dims = 12;

struct Plant {
    double mass = 1.0;
    Matrix3d inertia = Matrix3d::Identity();
};

struct StateControl {
    Matrix3dual matrix;
    const Plant& plant;

    template<typename T>
    explicit StateControl(const T& matrix, const Plant& plant);

    [[nodiscard]] Vector3dual angular_velocity() const;
    [[nodiscard]] auto dynamics() const;
    [[nodiscard]] auto state() const;

    [[nodiscard]] auto param() const;
    auto param();
};

template<typename T>
StateControl::StateControl(const T& matrix, const Plant& plant) :
        matrix(matrix),
        plant{plant} {
}

Vector3dual StateControl::angular_velocity() const {
    auto theta = matrix.col(2);
    auto theta_dt = matrix.col(3);

    auto cos_theta_y = cos(theta.y());
    auto sin_theta_y = sin(theta.y());
    auto cos_theta_z = cos(theta.z());
    auto sin_theta_z = sin(theta.z());
    Matrix3dual transform;
    transform << cos_theta_y * cos_theta_z, -sin_theta_z, 0,
            cos_theta_y * sin_theta_z, cos_theta_z, 0,
            -sin_theta_y, 0, 1;
    return transform * theta_dt;
}

auto StateControl::dynamics() const {
    auto r = matrix.col(0);
    auto r_dt = matrix.col(1);
    auto p = matrix.col(4);
    auto p_dt = matrix.col(5);
    auto f = matrix.col(6);

    const Vector3d gravity(0, 9.8, 0);
    auto linear_acc = f - plant.mass * gravity;
    auto angular_vel = angular_velocity();
    auto angular_acc = f.cross(r - p) - angular_vel.cross(plant.inertia * angular_vel);
    return (Matrix<dual, dynamics_dims, 1>() << r_dt, linear_acc, angular_acc, p_dt).finished();
}

auto StateControl::state() const {
    auto r = matrix.col(0);
    auto r_dt = matrix.col(1);
    auto angular_vel = angular_velocity();
    auto p = matrix.col(4);
    return (Matrix<dual, dynamics_dims, 1>() << r, r_dt, angular_vel, p).finished();
}

auto StateControl::param() const {
    return Map<const Matrix<dual, state_dims, 1>>(matrix.data());
}

auto StateControl::param() {
    return Map<Matrix<dual, state_dims, 1>>(matrix.data());
}

template<typename T>
auto chip_tensor(const T& tensor, int index) {
    TensorRef<Tensor<const double, 2>> chip = tensor.chip(index, 0);
    Map<const MatrixXd> matrix(chip.data(), chip.dimension(1), chip.dimension(2));
    return matrix;
}

void collocation(unsigned output_dims,
                 double* output,
                 unsigned input_dims,
                 const double* input,
                 double* grad,
                 void* data) {
    auto& plant = *reinterpret_cast<Plant*>(data);

    auto horizon = input_dims / state_dims;
    TensorMap<Tensor<const double, 3>> input_tensor(input, horizon, 3, state_dims / 3);
    Map<MatrixXd> output_matrix(output, output_dims / dynamics_dims, dynamics_dims);
    Map<MatrixXd> grad_matrix(grad, output_dims, input_dims);

    for (int i = 0; i < horizon - 1; ++i) {
        StateControl current(chip_tensor(input_tensor, i), plant);
        StateControl next(chip_tensor(input_tensor, i + 1), plant);

        auto func = [&] {
            auto delta = next.state() - current.state();
            auto sum = next.dynamics() + current.dynamics();
            return delta - 0.5 * delta_seconds * sum;
        };

        VectorXdual output_dual;
        if (grad != nullptr) {
            constexpr auto dims = state_dims * 2;
            auto start_row = dynamics_dims * i;
            auto start_col = state_dims * i;
            auto block = grad_matrix.block<dynamics_dims, dims>(start_row, start_col);
            block << jacobian(func, wrt(current.param(), next.param()), at(), output_dual);
        } else output_dual = func();
        output_matrix.col(i) << output_dual.cast<double>();
    }
}

int main() {
    return 0;
}