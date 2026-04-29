# lmfit

A header-only C++ template library for non-linear least squares problems using the Levenberg-Marquardt (LM) algorithm.

## Requirements

* **C++17** or later.
* **Eigen** (Linear algebra library).

## Usage

### 1. Define your Problem Class

You must provide a `Problem` class that implements the following interface:

```cpp
#include <iostream>
#include <Eigen/Dense>

#include "lm.hpp"

struct ExponentialModel
{
	Eigen::VectorXd t;

	// y = a * exp(-b * t) + c
	Eigen::VectorXd forward(const Eigen::VectorXd& x) const
	{
		return x(0) * (-x(1) * t).array().exp() + x(2);
	}

	std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& x) const
	{
		Eigen::VectorXd y = forward(x);
		Eigen::MatrixXd J(t.size(), 3);
		
		Eigen::VectorXd e_bt = (-x(1) * t).array().exp();
		
		J.col(0) = e_bt; // dy/da
		J.col(1) = -x(0) * t.array() * e_bt.array(); // dy/db
		J.col(2).setConstant(1.0); // dy/dc
		
		return {y, J};
	}

	Eigen::VectorXd constrain(const Eigen::VectorXd& x) const
	{
		Eigen::VectorXd x_res = x;
		if (x_res(1) < 0.0) x_res(1) = 0.0;
		return x_res;
	}
};

struct Gaussian2DModel
{
	Eigen::VectorXd grid_x;
	Eigen::VectorXd grid_y;

	// z = A * exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))
	Eigen::VectorXd forward(const Eigen::VectorXd& p) const
	{
		double A = p(0), x0 = p(1), y0 = p(2), sigma = p(3), B = p(4);
		
		Eigen::VectorXd r2 = (grid_x.array() - x0).pow(2) + (grid_y.array() - y0).pow(2);
		
		return (A * (-r2.array() / (2.0 * sigma * sigma)).exp() + B).matrix();
	}

	std::pair<Eigen::VectorXd, Eigen::MatrixXd> evaluate(const Eigen::VectorXd& p) const
	{
		int n = grid_x.size();
		double A = p(0), x0 = p(1), y0 = p(2), sigma = p(3), B = p(4);

		Eigen::VectorXd z = forward(p);
		Eigen::MatrixXd J(n, 5);

		Eigen::VectorXd r2 = (grid_x.array() - x0).pow(2) + (grid_y.array() - y0).pow(2);
		Eigen::VectorXd exp_part = (-r2.array() / (2.0 * sigma * sigma)).exp().matrix();

		J.col(0) = exp_part; // dz/dA
		J.col(1) = (A / (sigma * sigma)) * (grid_x.array() - x0) * exp_part.array(); // dz/dx0
		J.col(2) = (A / (sigma * sigma)) * (grid_y.array() - y0) * exp_part.array(); // dz/dy0
		J.col(3) = (A / (sigma * sigma * sigma)) * r2.array() * exp_part.array(); // dz/dsigma
		J.col(4).setConstant(1.0); // dz/dB

		return {z, J};
	}

	Eigen::VectorXd constrain(const Eigen::VectorXd& p) const
	{
		Eigen::VectorXd p_new = p;
		if (p_new(3) < 1e-3) p_new(3) = 1e-3; // sigma > 0
		return p_new;
	}
};

int main()
{
    {
		int n = 500;
		Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n, 0, 5);
		
		// answer [a=5.0, b=0.8, c=1.2] + noise
		Eigen::VectorXd y_obs = 5.0 * (-0.8 * t).array().exp() + 1.2;
		y_obs += Eigen::VectorXd::Random(n) * 0.4;

		ExponentialModel model{t};
		LevenbergMarquardt<ExponentialModel> solver;
		
		Eigen::VectorXd x_init(3);
		x_init << 10.0, 0.01, 0.1; // initial guess

		auto res = solver.minimize(model, y_obs, x_init);

		std::cout << "--- 1D Fitting Result ---" << std::endl;
		std::cout << "y = a * exp(- b * x) + c" << std::endl;
		std::cout << "a: " << res.x(0) << " +/- " << res.x_error(0) << std::endl;
		std::cout << "b: " << res.x(1) << " +/- " << res.x_error(1) << std::endl;
		std::cout << "c: " << res.x(2) << " +/- " << res.x_error(2) << std::endl;
	}

	{
		const int nx = 100, ny = 100;
		Eigen::VectorXd gx(nx * ny), gy(nx * ny);
		
		for (int i = 0; i < nx; ++i)
		{
			for (int j = 0; j < ny; ++j)
			{
				gx(i * ny + j) = i * 0.5;
				gy(i * ny + j) = j * 0.5;
			}
		}

		// answer [A=10, x0=2.5, y0=2.5, sigma=1.0, B=1.0] + noise
		Gaussian2DModel model{gx, gy};
		Eigen::VectorXd p_true(5);
		p_true << 10.0, 2.5, 2.5, 1.0, 1.0;
		Eigen::VectorXd z_obs = model.forward(p_true) + Eigen::VectorXd::Random(nx * ny) * 0.1;

		LevenbergMarquardt<Gaussian2DModel> solver;
		Eigen::VectorXd p_init(5);
		p_init << 5.0, 1.0, 1.0, 2.0, 0.0; // initial guess

		auto res = solver.minimize(model, z_obs, p_init);

		std::cout << "--- 2D Fitting Result ---" << std::endl;
		std::cout << "z = A * exp(-((x - x0)^2 + (y - y0)^2) / (2 * sigma^2))" << std::endl;
		std::cout << "A    : " << res.x(0) << " +/- " << res.x_error(0) << std::endl;
		std::cout << "x0   : " << res.x(1) << " +/- " << res.x_error(1) << std::endl;
		std::cout << "y0   : " << res.x(2) << " +/- " << res.x_error(2) << std::endl;
		std::cout << "sigma: " << res.x(3) << " +/- " << res.x_error(3) << std::endl;
		std::cout << "B    : " << res.x(4) << " +/- " << res.x_error(4) << std::endl;
	}

    return 0;
}
