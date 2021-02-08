// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

/// Information about mappings.

namespace basix::mapping
{

/// Cell type
enum class type
{
  identity,
  covariantPiola,
  contravariantPiola,
  doubleCovariantPiola,
  doubleContravariantPiola,
};

/// Get the function that applies the forward map.
/// The inputs of the returned function are:
/// - [in/out] The data to apply the mapping to
/// - [in] The Jacobian
/// - [in] detJ The determinant of the Jacobian
/// - [in] K The inverse of the Jacobian
/// @param[in] mapping_type Mapping type
/// @param value_shape The value shape of the data
/// @return The function
// TODO: should data be in/out?
template <typename T>
std::function<std::vector<T>(const std::vector<T>&, const std::vector<double>&,
                             const double, const std::vector<double>&)>
get_forward_map(
    // const Eigen::Array<T, Eigen::Dynamic, 1>& reference_data,
    //                 const Eigen::MatrixXd& J, double detJ,
    //                 const Eigen::MatrixXd& K,
    mapping::type mapping_type, const std::vector<int> value_shape)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return
        [](const std::vector<T>& reference_data, const std::vector<double>&,
           const double, const std::vector<double>&) { return reference_data; };
  case mapping::type::covariantPiola:
    return [](const std::vector<T>& reference_data, const std::vector<double>&,
              const double, const std::vector<double>& K) {
      const int tdim = reference_data.size();
      const int gdim = K.size() / reference_data.size();
      std::vector<T> result(gdim);
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> M(
          K.data(), gdim, tdim);
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> V(
          reference_data.data(), tdim);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(result.data(), gdim)
          = M * V;
      return result;
    };
  case mapping::type::contravariantPiola:
    return
        [](const std::vector<T>& reference_data, const std::vector<double>& J,
           const double detJ, const std::vector<double>&) {
          const int tdim = reference_data.size();
          const int gdim = J.size() / reference_data.size();
          std::vector<T> result(gdim);
          Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>
              M(J.data(), gdim, tdim);
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> V(
              reference_data.data(), tdim);
          Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(result.data(), gdim)
              = 1 / detJ * M * V;
          return result;
        };
  case mapping::type::doubleCovariantPiola:
  {
    assert(value_shape.size() == 2);
    assert(value_shape[0] == value_shape[1]);
    if (value_shape[0] == 2)
    {
      return [](const std::vector<T>& reference_data,
                const std::vector<double>&, const double,
                const std::vector<double>& K) {
        const int gdim = K.size() / 2;
        std::vector<T> result(gdim * gdim);
        Eigen::Map<
            const Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>>
            M(K.data(), 2, gdim);
        Eigen::Map<const Eigen::Matrix<T, 2, 2>> V(reference_data.data());
        Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            result.data(), gdim, gdim)
            = M.transpose() * V * M;
        return result;
      };
    }
    else
    {
      assert(value_shape[0] == 3);
      return [](const std::vector<T>& reference_data,
                const std::vector<double>&, const double,
                const std::vector<double>& K) {
        const int gdim = K.size() / 3;
        std::vector<T> result(gdim * gdim);
        Eigen::Map<
            const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
            M(K.data(), 3, gdim);
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> V(reference_data.data());
        Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            result.data(), gdim, gdim)
            = M.transpose() * V * M;
        return result;
      };
    }
  }
  case mapping::type::doubleContravariantPiola:
  {
    assert(value_shape.size() == 2);
    assert(value_shape[0] == value_shape[1]);
    if (value_shape[0] == 2)
    {
      return [](const std::vector<T>& reference_data,
                const std::vector<double>& J, const double detJ,
                const std::vector<double>&) {
        const int gdim = J.size() / 2;
        std::vector<T> result(gdim * gdim);
        Eigen::Map<
            const Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>>
            M(J.data(), 2, gdim);
        Eigen::Map<const Eigen::Matrix<T, 2, 2>> V(reference_data.data());
        Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            result.data(), gdim, gdim)
            = 1 / (detJ * detJ) * M * V * M.transpose();
        return result;
      };
    }
    else
    {
      assert(value_shape[0] == 3);
      return [](const std::vector<T>& reference_data,
                const std::vector<double>& J, const double detJ,
                const std::vector<double>&) {
        const int gdim = J.size() / 3;
        std::vector<T> result(gdim * gdim);
        Eigen::Map<
            const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
            M(J.data(), 3, gdim);
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> V(reference_data.data());
        Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            result.data(), gdim, gdim)
            = 1 / (detJ * detJ) * M * V * M.transpose();
        return result;
      };
    }
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>
map_push_forward(const Eigen::Array<T, Eigen::Dynamic, 1>& reference_data,
                 const Eigen::MatrixXd& J, double detJ,
                 const Eigen::MatrixXd& K, mapping::type mapping_type,
                 const std::vector<int> value_shape)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return reference_data;
  case mapping::type::covariantPiola:
    return K.transpose() * reference_data.matrix();
  case mapping::type::contravariantPiola:
    return 1 / detJ * J * reference_data.matrix();
  case mapping::type::doubleCovariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = K.transpose() * data_matrix * K;
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(
        result.data(), reference_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = 1 / (detJ * detJ) * J * data_matrix * J.transpose();
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(
        result.data(), reference_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

/// Apply inverse mapping
/// @param physical_data The data to apply the inverse mapping to
/// @param J The Jacobian
/// @param detJ The determinant of the Jacobian
/// @param K The inverse of the Jacobian
/// @param mapping_type Mapping type
/// @param value_shape The value shape of the data
/// @return The mapped data
// TODO: should data be in/out?
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>
map_pull_back(const Eigen::Array<T, Eigen::Dynamic, 1>& physical_data,
              const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K,
              mapping::type mapping_type, const std::vector<int> value_shape)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return physical_data;
  case mapping::type::covariantPiola:
    return J.transpose() * physical_data.matrix();
  case mapping::type::contravariantPiola:
    return detJ * K * physical_data.matrix();
  case mapping::type::doubleCovariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = J.transpose() * data_matrix * J;
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(result.data(),
                                                          physical_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = detJ * detJ * K * data_matrix * K.transpose();
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(result.data(),
                                                          physical_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

/// Convert mapping type enum to string
inline const std::string& type_to_str(mapping::type type)
{
  static const std::map<mapping::type, std::string> type_to_name = {
      {mapping::type::identity, "identity"},
      {mapping::type::covariantPiola, "covariant Piola"},
      {mapping::type::contravariantPiola, "contravariant Piola"},
      {mapping::type::doubleCovariantPiola, "double covariant Piola"},
      {mapping::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}

} // namespace basix::mapping
