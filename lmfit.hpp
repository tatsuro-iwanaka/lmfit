#include <iostream>
#include <Eigen/Dense>

namespace lmfit
{

template <typename Problem> class LMFit
{
	public:
		struct Config
		{
			int max_iter = 100;
			double tol = 1e-6;   // 残差の相対変化
			double tol_x = 1e-8; // パラメータの更新幅
			double tol_g = 1e-8; // 勾配の大きさ
			double lambda_init = 1e-3;
			double lambda_factor = 10.0;
		};

		struct Result
		{
			Eigen::VectorXd x;        // 最適化されたパラメータ
			Eigen::VectorXd x_error;  // 各パラメータの標準誤差 (1-sigma)
			double final_chi2;        // 最終的な残差平方和
			bool converged;           // 収束したか
		};

		explicit LMFit(Config config = {}) : config_(std::move(config))
		{
			;
		}

		Result minimize(Problem&, const Eigen::VectorXd&, const Eigen::VectorXd&) const;

	private:
		Config config_;

		Eigen::VectorXd calculateError(const Eigen::MatrixXd&, double, int, int) const;
};

template <typename Problem> inline LMFit<Problem>::Result LMFit<Problem>::minimize(Problem& problem, const Eigen::VectorXd& y_obs, const Eigen::VectorXd& x_init) const
{
	Eigen::VectorXd x = x_init;
	double lambda = config_.lambda_init;
	const int n_obs = y_obs.size();
	const int n_param = x.size();

	auto [y_calc, J] = problem.evaluate(x);
	Eigen::VectorXd residual = y_obs - y_calc;
	double chi2 = residual.squaredNorm();

	for (int iter = 0; iter < config_.max_iter; ++iter)
	{
		Eigen::MatrixXd AtA = J.transpose() * J;
		Eigen::VectorXd Atb = J.transpose() * residual;

		if (Atb.lpNorm<Eigen::Infinity>() < config_.tol_g)
		{
			return {x, calculateError(J, chi2, n_obs, n_param), chi2, true};
		}

		for (int i = 0; i < n_param; ++i)
		{
			AtA(i, i) += lambda * (AtA(i, i) + 1e-9); 
		}

		Eigen::VectorXd dx = AtA.ldlt().solve(Atb);

		if (dx.norm() < config_.tol_x * (x.norm() + config_.tol_x))
		{
			return {x, calculateError(J, chi2, n_obs, n_param), chi2, true};
		}

		Eigen::VectorXd x_new = problem.constrain(x + dx);
		Eigen::VectorXd y_new = problem.forward(x_new);
		double chi2_new = (y_obs - y_new).squaredNorm();

		if (chi2_new < chi2)
		{
			bool converged = (chi2 - chi2_new) / chi2 < config_.tol;

			auto [y_next, J_next] = problem.evaluate(x_new);
			x = x_new;
			y_calc = std::move(y_next);
			J = std::move(J_next);
			residual = y_obs - y_calc;
			chi2 = chi2_new;
			lambda /= config_.lambda_factor;

			if (converged)
			{
				break;
			}
		}
		else
		{
			lambda *= config_.lambda_factor;
		}

		if (lambda > 1e12) break;
	}

	return {x, calculateError(J, chi2, n_obs, n_param), chi2, false};
}

template <typename Problem> inline Eigen::VectorXd LMFit<Problem>::calculateError(const Eigen::MatrixXd& J, double chi2, int n_obs, int n_param) const
{
	double dof = std::max(1.0, static_cast<double>(n_obs - n_param));
	double sigma2 = chi2 / dof;

	Eigen::MatrixXd AtA = J.transpose() * J;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(AtA, Eigen::ComputeFullU | Eigen::ComputeFullV);
	
	double tolerance = std::numeric_limits<double>::epsilon() * std::max(AtA.rows(), AtA.cols()) * svd.singularValues().array().abs()(0);
	auto inv_singular_values = svd.singularValues().array().unaryExpr([&](double s){return (s > tolerance) ? 1.0 / s : 0.0;});

	Eigen::MatrixXd cov = svd.matrixV() * inv_singular_values.matrix().asDiagonal() * svd.matrixU().transpose();

	return cov.diagonal().cwiseAbs().cwiseSqrt() * std::sqrt(sigma2);
}

}
