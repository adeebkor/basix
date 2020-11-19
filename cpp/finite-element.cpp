// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial-set.h"
#include <iostream>
#include <map>

using namespace libtab;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    std::string family_name, cell::Type cell_type, int degree,
    const std::vector<int>& value_shape, const Eigen::ArrayXXd& coeffs,
    const std::vector<std::vector<int>>& entity_dofs,
    const std::vector<Eigen::MatrixXd>& base_permutations)
    : _cell_type(cell_type), _degree(degree), _value_shape(value_shape),
      _coeffs(coeffs), _entity_dofs(entity_dofs),
      _base_permutations(base_permutations), _family_name(family_name)
{
  // Check that entity dofs add up to total number of dofs
  int sum = 0;
  for (const std::vector<int>& q : entity_dofs)
  {
    for (const int& w : q)
      sum += w;
  }

  if (sum != _coeffs.rows())
  {
    throw std::runtime_error(
        "Number of entity dofs does not match total number of dofs");
  }
}
//-----------------------------------------------------------------------------
cell::Type FiniteElement::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
int FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const
{
  int value_size = 1;
  for (const int& d : _value_shape)
    value_size *= d;
  return value_size;
}
//-----------------------------------------------------------------------------
const std::vector<int>& FiniteElement::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::ndofs() const { return _coeffs.rows(); }
//-----------------------------------------------------------------------------
std::string FiniteElement::family_name() const { return _family_name; }
//-----------------------------------------------------------------------------
std::vector<std::vector<int>> FiniteElement::entity_dofs() const
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd
FiniteElement::compute_expansion_coefficients(const Eigen::MatrixXd& coeffs,
                                              const Eigen::MatrixXd& dual,
                                              bool condition_check)
{
#ifndef NDEBUG
  std::cout << "Initial coeffs = \n[" << coeffs << "]\n";
  std::cout << "Dual matrix = \n[" << dual << "]\n";
#endif

  const Eigen::MatrixXd A = coeffs * dual.transpose();
  if (condition_check)
  {
    Eigen::JacobiSVD svd(A);
    const int size = svd.singularValues().size();
    const double kappa
        = svd.singularValues()(0) / svd.singularValues()(size - 1);
    if (kappa > 1e6)
    {
      throw std::runtime_error("Poorly conditioned B.D^T when computing "
                               "expansion coefficients");
    }
  }

  Eigen::MatrixXd new_coeffs = A.colPivHouseholderQr().solve(coeffs);
#ifndef NDEBUG
  std::cout << "New coeffs = \n[" << new_coeffs << "]\n";
#endif

  return new_coeffs;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd>
FiniteElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  const int tdim = cell::topological_dimension(_cell_type);
  if (x.cols() != tdim)
    throw std::runtime_error("Point dim does not match element dim.");

  std::vector<Eigen::ArrayXXd> basis
      = polyset::tabulate(_cell_type, _degree, nd, x);
  const int psize = polyset::size(_cell_type, _degree);
  const int ndofs = _coeffs.rows();
  const int vs = value_size();

  std::vector<Eigen::ArrayXXd> dresult(basis.size(),
                                       Eigen::ArrayXXd(x.rows(), ndofs * vs));
  for (std::size_t p = 0; p < dresult.size(); ++p)
  {
    for (int j = 0; j < vs; ++j)
    {
      dresult[p].block(0, ndofs * j, x.rows(), ndofs)
          = basis[p].matrix()
            * _coeffs.block(0, psize * j, _coeffs.rows(), psize).transpose();
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::MatrixXd> FiniteElement::base_permutations() const
{
  return _base_permutations;
};
//-----------------------------------------------------------------------------
