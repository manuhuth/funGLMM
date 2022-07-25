using Pkg

#Pkg.add(["ProgressMeter"])

using Random
using Base.Threads
using BSplines
using BenchmarkTools
using XLSX
using DataFrames
using QuadGK
using Plots
using StaticArrays
using CategoricalArrays
using Distributions
using FastGaussQuadrature
using Profile
using LinearAlgebra
using Base.Threads
using ForwardDiff
using Optim
using StatsBase
using FreqTables
using LineSearches
using ProgressMeter


#---------------functions only used for development-----------------------------
function get_dummy_beta_parameters_development(length_betas)
  betas = Vector{Float64}()
  for i in eachindex(length_betas)
    append!(betas, rand(Uniform(-20, 20), length_betas[i]))#this will come from the function later
  end
  return betas #maybe a good starting value function
end

function compute_random_starting_values_theta(family, link, ordinal_values_low_to_high, functional_covariates, basis_type_weighting_betas, scalar_covariates)

  length_betas_all = compute_beta_basis_lengths(functional_covariates, basis_type_weighting_betas)
  length_alphas = get_length_alphas(family, link, ordinal_values_low_to_high)

  betas = get_dummy_beta_parameters_development(length_betas_all) #this will come from the function optimization
  gammas = rand(Uniform(-1, 1), length(scalar_covariates))  #this will come from the function optimization
  alphas = rand(Uniform(log(0.2), log(3)), length_alphas) #this will come from the function optimization
  transformed_sigma_mu = log.(rand(Uniform(0.3, 4), 1)) #this will come from the function optimization
  theta = vcat(betas, gammas, alphas, transformed_sigma_mu)

  return theta
end


#---------------Package functions-----------------------------------------------



#---------------Array sums------------------------------------------------------
function compute_sum_offset_array(offset_array, array)
  """
  Computes the inner vector product for two arrays that can potentially be
  offset arrays.

  # Arguments
  - `offsetArray::offset_array`: an offset vector, also works with ordinary vectors.
  - `array::array`: an array.
  ...
  """
  sum = 0.0

  for i in eachindex(offset_array)
    sum = sum + offset_array[i] * array[i]
  end
  return(sum)
end

function compute_inner_vector_product(y, x)
  out = 0.0
  for i in 1:length(y)
    out = y[i]*x[i] + out
  end
  return out
end

#---------------Mathematical base functions-------------------------------------
function logistic_function(x)
  """
  Computes value of the logistic function.

  # Arguments
  - `Float::x`: value at which the logistic function should be evaluated.
  ...
  """
  return 1 ./ (1 .+ exp.(-x))
end

function identity_function(x)
  """
  Wrapper function for the identity function. Used as link function.

  # Arguments
  - `float::x`: function input.
  ...
  """
  return x
end

function natural_logarithm_function(x)
  """
  Wrapper function for the natural logarithm function. Used as link function.

  # Arguments
  - `float::x`: function input.
  ...
  """
  return log.(x)
end

function exponential_function(x)
  """
  Wrapper function for the exponential function. Used as link function.

  # Arguments
  - `float::x`: function input.
  ...
  """
  return exp.(x)
end

function derivative_logistic_alpha_Z(alpha, Z)
  return - (1 / (1 + exp(-alpha+Z)))^2 * exp(-alpha+Z)
end

function derivative_logistic(x)
  return - (1 / (1 + exp(x)))^2 * exp(-x)
end

function constant_one(x)
  return 1
end
#------------------basis function related---------------------------------------
function compute_linear_basis_term(t, parameter, basis)
  """
  Computes value of the logistic function.

  # Arguments
  - `Float::t`: value at which the spline should be evaluated.
  - `Vector::parameter`: vector of basis coefficients.
  - `basisObject::basis`: julia basis object.
  ...
  """

  if check_if_basis_is_BSpline(basis)
    return compute_sum_offset_array(bsplines(basis, t), parameter)
  end #end if of BSPlines
end

#------------------mathematical operations--------------------------------------
function compute_ols_vector(y, x)
  """
  Computes the ordinary least squares estimate for a given feature matrix x
  and output vector y. If a constant is desired it must be already in x.

  # Arguments
  - `array::y`: output vector.
  - `array::x`: feature matrix.
  ...
  """
  return inv(x'*x)*x'*y
end

#-------------------test data related-------------------------------------------
function get_data_spain(path="/home/manuel/Documents/fun_glm/test_data/containment_1_Spain.xlsx",
  name_y = "containment_index", name_time = "time")
  """
  Loads the COVID-19 containment index data for Spain.

  # Arguments
  - `str::path`: path where xlsx data is stored.
  - `str::name_y`: name of containment index column.
  - `str::name_t`: name of time column.
  ...
  """
  df = DataFrame(XLSX.readtable(path, 1)...)
  y = convert(Array{Float64,1}, df[!, name_y])
  time = convert(Array{Float64,1},df[!, name_time])

  return(y, time)
end

function compute_parameter_and_basis_data_spain(basis_order, basis_breakpoints,
                                                return_fitted=false)
  """
  Computes the parameter and and the basis for the data for Spain. If desired,
  the fit of the data is also computed.

  # Arguments
  - `int::basis_order`: Order of the basis. See Julia BSplines.
  - `array::basis_breakpoints`: Breakpoints of the splines. See Julia BSplines.
  - `bool::return_fitted`: true if smoothed data curve should be returned as third
                            output.
  ...
  """
  y, time = get_data_spain()
  y_log = log.(y) #logarithm of y; need to be retransformed in compute_integral_term_s_t
  basis_data = BSplineBasis(basis_order, basis_breakpoints)
  return compute_BSpline_basis_and_coefficients(y_log, time, basis_order, basis_breakpoints,
                                                return_fitted)
end

function load_functional_test_data_dict(path, countries, basis_order, basis_breakpoints,
                        link_function_data::Function=identity_function,
                        number=1, name_time =  "time", name_y="containment_index",
                        data_frames = true
                        )

  dict_containment = Dict()
  for j in 1:number
    dict_subgroups = Dict()
    for i in eachindex(countries)
      df = DataFrame(XLSX.readtable(path * "containment_" * "$j" * "_" * countries[i] * ".xlsx", 1)...)
      y = convert(Array{Float64,1},df[!, name_y])
      y_link = link_function_data.(y)
      time = convert(Array{Float64,1},df[!, name_time])
      parameter, basis, fit = compute_BSpline_basis_and_coefficients(y_link, time, basis_order, basis_breakpoints, true)
      if data_frames
        dict_subgroups[countries[i] * "_$j"] = Dict("data_frame" => df, "basis" => basis,
                                                    "basis_parameter" => parameter, "fit" => fit,
                                                    "link_function" => exponential_function,
                                                    "betas_sim" => rand(Uniform(-4, 4), length(basis)))
      else
        dict_subgroups[countries[i] * "_$j"] = Dict("basis" => basis,
                                                    "basis_parameter" => parameter,
                                                    "fit" => fit,
                                                    "link_function" => exponential_function,
                                                    "betas_sim" => rand(Uniform(-10, 10), length(basis)))
      end
    dict_containment["containment_$j"] = dict_subgroups
    end
  end
  return dict_containment
end

function load_scalar_test_data(path)
    return DataFrame(XLSX.readtable(path, 1, infer_eltypes=true)...)
end


#---------------------------b_spline basis related------------------------------
function compute_BSpline_basis_and_coefficients(y, time, basis_order, basis_breakpoints,
                                                return_fitted=false)
  """
  Computes the parameter and and the basis for the data for Spain. If desired,
  the fit of the data is also computed.

  # Arguments
  - `array::y`: observations over time
  - `array::time`: time at which observations are taken.
  - `int::basis_order`: Order of the basis. See Julia BSplines.
  - `array::basis_breakpoints`: Breakpoints of the splines. See Julia BSplines.
  - `bool::return_fitted`: true if smoothed data curve should be returned as third
                            output.
  ...
  """
  basis_data = BSplineBasis(basis_order, basis_breakpoints)
  B = basismatrix(basis_data, time)
  parameter = compute_ols_vector(y, B)

  if !return_fitted
    return parameter, basis_data
  else
    fit_y = B * parameter
    return parameter, basis_data, fit_y
  end
end

function check_if_basis_is_BSpline(basis)
  if typeof(basis) <: BSplineBasis{}
    return true
  else
    return false
  end
end

#---------------------------Z and the functional integral term-------------------
function compute_functional_integral_term_s_t(s, t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta,
  link_function_data::Function, link_function_beta::Function)
  """
  Computes the integral term within Z, which is the product of the data function
  and the transformed beta function.

  # Arguments
  - `float::s`: point in time for which the integral term should be evaluated.
  - `float::t`: point in time at which the observation was taken.
  - `float::memory`: length of memory. Used to scale the domain of the beta function.
  - `array::coefficients_data`: coefficients of the data basis functions.
  - `BSplineBasis::basis_data`: data basis.
  - `array::coefficients_beta`: coefficients of the beta basis functions.
  - `BSplineBasis::basis_beta`: beta basis.
  - `function::link_function_data`: link function used for the data. Useful
                                    to retransform the data after it was
                                    transformed to ensure for example
                                    non-negativity.
  - `function::link_function_beta`: link function used for beta. Useful
                                    to transform the weights to a desired range.
  ...
  """

  return link_function_data(compute_linear_basis_term(s, coefficients_data, basis_data)) *
         link_function_beta(compute_linear_basis_term(s-t+memory, coefficients_beta, basis_beta))
end

function compute_functional_integral_term_s_t_derivative_j(j, s, t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta,
  link_function_data::Function,
  link_function_derivative_beta::Function)
  """
  Computes the integral term within Z, which is the product of the data function
  and the transformed beta function.

  # Arguments
  - `float::s`: point in time for which the integral term should be evaluated.
  - `float::t`: point in time at which the observation was taken.
  - `float::memory`: length of memory. Used to scale the domain of the beta function.
  - `array::coefficients_data`: coefficients of the data basis functions.
  - `BSplineBasis::basis_data`: data basis.
  - `array::coefficients_beta`: coefficients of the beta basis functions.
  - `BSplineBasis::basis_beta`: beta basis.
  - `function::link_function_data`: link function used for the data. Useful
                                    to retransform the data after it was
                                    transformed to ensure for example
                                    non-negativity.
  - `function::link_function_derivative_beta`: derivative of link function used
                                               for beta. Useful
                                               to transform the weights to a desired range.
  ...
  """
  unit_vector_j = I[1:length(coefficients_beta), j] * 1.0
  return link_function_data(compute_linear_basis_term(s, coefficients_data, basis_data)) *
         link_function_derivative_beta(compute_linear_basis_term(s-t+memory, coefficients_beta, basis_beta)) *
         compute_linear_basis_term(s-t+memory, unit_vector_j, basis_beta)
end

function compute_functional_integral_term(t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta, mini_time, maxi_time,
  link_function_data::Function, link_function_beta::Function, gauss_legendre_points, w_gauss_legendre)
  """
  Wrapper function for compute_integral_term_s_t. Computes the integral
  term within Z, which is the product of the data function
  and the transformed beta function and integrates it. Integration over s from
  the compute_integral_term_s_t function.

  # Arguments
  - `float::t`: point in time at which the observation was taken.
  - `float::memory`: length of memory. Used to scale the domain of the beta function.
  - `array::coefficients_data`: coefficients of the data basis functions.
  - `BSplineBasis::basis_data`: data basis.
  - `array::coefficients_beta`: coefficients of the beta basis functions.
  - `BSplineBasis::basis_beta`: beta basis.
  - `function::link_function_data`: link function used for the data. Useful
                                    to retransform the data after it was
                                    transformed to ensure for example
                                    non-negativity.
  - `function::link_function_beta`: link function used for beta. Useful
                                    to transform the weights to a desired range.
  ...
  """

  lower, upper = compute_lower_upper_bounds_integral(t, memory, mini_time, maxi_time)

  return compute_integral_gauss_legendre(t, memory, coefficients_data, basis_data,
    coefficients_beta, basis_beta, lower, upper,
    link_function_data::Function, link_function_beta::Function, gauss_legendre_points, w_gauss_legendre)
end

function compute_integral_gauss_legendre(t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta, lower, upper,
  link_function_data::Function, link_function_beta::Function, gauss_legendre_points, w_gauss_legendre)

  sum = 0.0
  for i in 1:length(gauss_legendre_points)
    sum = sum + w_gauss_legendre[i]*compute_functional_integral_term_s_t((upper-lower)/2*gauss_legendre_points[i]+(upper+lower)/2,
                                                                        t, memory, coefficients_data, basis_data,
                                                                        coefficients_beta, basis_beta,
                                                                        link_function_data, link_function_beta)
  end

  return (upper - lower) / 2 * sum
end

function compute_integral_derivative_gauss_legendre_j(j, t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta, lower, upper,
  link_function_data::Function, link_function_derivative_beta::Function,
  gauss_legendre_points, w_gauss_legendre)

  sum = 0.0
  for i in 1:length(gauss_legendre_points)
    sum = sum + w_gauss_legendre[i]*compute_functional_integral_term_s_t_derivative_j(j, (upper-lower)/2*gauss_legendre_points[i]+(upper+lower)/2,
                                                                        t, memory, coefficients_data, basis_data,
                                                                        coefficients_beta, basis_beta,
                                                                        link_function_data, link_function_derivative_beta)
  end

  return (upper - lower) / 2 * sum
end

function compute_functional_integral_term_derivative_j(j, t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta, mini_time, maxi_time,
  link_function_data::Function, link_function_derivative_beta::Function, gauss_legendre_points, w_gauss_legendre)
  """
  Wrapper function for compute_integral_term_s_t. Computes the integral
  term within Z, which is the product of the data function
  and the transformed beta function and integrates it. Integration over s from
  the compute_integral_term_s_t function.

  # Arguments
  - `float::t`: point in time at which the observation was taken.
  - `float::memory`: length of memory. Used to scale the domain of the beta function.
  - `array::coefficients_data`: coefficients of the data basis functions.
  - `BSplineBasis::basis_data`: data basis.
  - `array::coefficients_beta`: coefficients of the beta basis functions.
  - `BSplineBasis::basis_beta`: beta basis.
  - `function::link_function_data`: link function used for the data. Useful
                                    to retransform the data after it was
                                    transformed to ensure for example
                                    non-negativity.
  - `function::link_function_beta`: link function used for beta. Useful
                                    to transform the weights to a desired range.
  ...
  """

  lower, upper = compute_lower_upper_bounds_integral(t, memory, mini_time, maxi_time)

  return compute_integral_derivative_gauss_legendre_j(j, t, memory, coefficients_data, basis_data,
    coefficients_beta, basis_beta, lower, upper,
    link_function_data::Function, link_function_derivative_beta::Function,
    gauss_legendre_points, w_gauss_legendre)
end

function compute_integral_quadgk(t, memory, coefficients_data, basis_data,
  coefficients_beta, basis_beta, lower, upper,
  link_function_data::Function, link_function_beta::Function)

  intf(t, memory, coefficients_data, basis_data,
    coefficients_beta, basis_beta,
    link_function_data::Function, link_function_beta::Function) =
     quadgk(s -> compute_functional_integral_term_s_t(s, t, memory, coefficients_data, basis_data,
       coefficients_beta, basis_beta,
       link_function_data::Function, link_function_beta::Function), lower, upper)

  integral_value_and_error = intf(t, memory, coefficients_data, basis_data,
    coefficients_beta, basis_beta,
    link_function_data, link_function_beta)
  return integral_value_and_error[1]
end

function compute_lower_upper_bounds_integral(t, memory, mini_time, maxi_time)
  if (t-memory) > mini_time && t < maxi_time
    return (t-memory), t
  elseif (t-memory) > mini_time && t >= maxi_time
    return t-memory, maxi_time
  elseif (t-memory) <= mini_time && t < maxi_time
    return mini_time, t
  else
    return mini_time, maxi_time
  end
end

function compute_sum_functional_integral_terms(data_grouped, individual_index,
                                              t_index, t_variable, basis_type_weighting_betas,
                                              beta_parameter_dict,
                                              functional_covariates, functional_data, name_no_functional_group,
                                              gauss_legendre_points, w_gauss_legendre)
  sum = 0.0
  for i in 1:length(functional_covariates)

    if data_grouped[individual_index][t_index, functional_covariates[i]] == name_no_functional_group
      continue
    end

    if functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["min_time"] >= data_grouped[individual_index][t_index, t_variable]
      continue
    elseif functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["max_time"] + basis_type_weighting_betas[functional_covariates[i]]["memory"] <= data_grouped[individual_index][t_index, t_variable]
      continue
    else
        sum = sum + compute_functional_integral_term(
        data_grouped[individual_index][t_index, t_variable],
        basis_type_weighting_betas[functional_covariates[i]]["memory"],
        functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["basis_parameter"],
        functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["basis"],
        beta_parameter_dict[functional_covariates[i]],
        basis_type_weighting_betas[functional_covariates[i]]["basis"],
        functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["min_time"],
        functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["max_time"],
        functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["link_function"],
        basis_type_weighting_betas[functional_covariates[i]]["link_function"],
        gauss_legendre_points, w_gauss_legendre
        )
    end
  end
  return sum
end

function compute_sum_functional_integrals_scalar_covariates_for_one_individual(data_grouped, individual_index,
                                              t_variable, basis_type_weighting_betas,
                                              beta_parameter_dict,
                                              functional_covariates, functional_data, gammas,
                                              name_no_functional_group,
                                              gauss_legendre_points, w_gauss_legendre, scalar_covariates,
                                              functional_integral_terms_plus_scalar_covariates = Array{Any}(undef, nrow((data_grouped[individual_index]))))

  for t_index in 1:nrow((data_grouped[individual_index]))
    matrix_scalar_covariates = Matrix(DataFrame(data_grouped[individual_index][t_index, scalar_covariates]))
    functional_integral_terms_plus_scalar_covariates[t_index] = compute_inner_vector_product(matrix_scalar_covariates, gammas) +
                                                          compute_sum_functional_integral_terms(data_grouped, individual_index,
                                                                                                        t_index, t_variable, basis_type_weighting_betas,
                                                                                                        beta_parameter_dict,
                                                                                                        functional_covariates, functional_data, name_no_functional_group,
                                                                                                         gauss_legendre_points, w_gauss_legendre)
  end
  return functional_integral_terms_plus_scalar_covariates
end

function compute_Z_i_t(mu_i, sum_functional_plus_scalar_covariates)
  """
  Computes Z_i_t as sum of the functional covariates and the scalar covariates.

  # Arguments
  - `float::mu_i`: value of the random intercept term.
  - `float::sum_functional_plus_scalar_covariates`: sum of all functional and scalar covariate terms.
  ...
  """
  return mu_i + sum_functional_plus_scalar_covariates
end

#----------------------------pre-processing ones--------------------------------
function add_min_max_time(functional_data_dictionary, t_variable)
  for functional_group in keys(functional_data_dictionary)
    for functional_subgroup in keys(functional_data_dictionary[functional_group])
      functional_data_dictionary[functional_group][functional_subgroup]["min_time"] = minimum(functional_data_dictionary[functional_group][functional_subgroup]["data_frame"][!,t_variable])
      functional_data_dictionary[functional_group][functional_subgroup]["max_time"] = maximum(functional_data_dictionary[functional_group][functional_subgroup]["data_frame"][!,t_variable])
    end
  end
  return functional_data_dictionary
end

function extract_dependent(formula)
    y = strip(split.(formula, "~")[1])
    return y
end

function extract_X_names(formula)
    X = map(strip, split.(split.(formula, "~")[2], "+"))

    functional_covariates = X[findall( x -> occursin("f(", x), X)]
    scalar_covariates = filter!(e->e∉functional_covariates,X)

    return chop.(functional_covariates, head=2, tail=1), scalar_covariates
end

function get_unique_values(df, name)
    unique_col = unique!(df[!, name])
    return unique_col
end

function sort_df(df, columns)
    return sort!(df[!, columns] )
end

function compute_beta_basis_lengths(functional_covariates, basis_type_weighting_betas)
  beta_length = Dict()
  for i in functional_covariates
    beta_length[i] = length(basis_type_weighting_betas[i]["basis"])
  end
  return beta_length
end

function get_length_alphas(family, link, ordinal_values_low_to_high)
  if family == "ordinal" && link == "logit"
    return  length(ordinal_values_low_to_high) - 1
  end
end

function get_y_calc(family, y_name)
  if family == "ordinal"
    return "y_cat"
  else
    return y_name
  end
end

function compute_gauss_legendre_penalty_points(basis_type_weighting_betas)
  dict_save = Dict()
  for i in keys(basis_type_weighting_betas)
    points, weights = gausslegendre(basis_type_weighting_betas[i]["gauss_legendre_points_penalty"])
    dict_save[i] = Dict("weights" => weights, "points" => points)
  end
  return dict_save
end
#-----------------------pre-processing each parameter ieration
function create_beta_dict(betas, length_betas)
    dict_betas = Dict()
    sum = 0
    for i in eachindex(length_betas)
      dict_betas[i] = betas[(sum+1):(length_betas[i]+sum)]
      sum = sum + length_betas[i]
    end
  return dict_betas
end



#---------------method specific regression------------------------------------
function create_numerical_ordinal_values(y, ordinal_values_low_to_high)
  y_cat = Array{Int}(undef, length(y))
  for i in 1:length(y)
      y_cat[i] = findall(x->x==y[i], ordinal_values_low_to_high)[1]
  end
  return y_cat
end

function probabilities_ordinal_logistic_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i, sum_functional_integrals_scalar_covariates_i_t)
  if y_i_t == 1
    return logistic_function(alphas[1] - compute_Z_i_t(mu_i, sum_functional_integrals_scalar_covariates_i_t))
  elseif y_i_t == length(ordinal_values_low_to_high)
    return 1 - logistic_function(alphas[y_i_t-1] - compute_Z_i_t(mu_i, sum_functional_integrals_scalar_covariates_i_t))
  else
    return logistic_function(alphas[y_i_t] - compute_Z_i_t(mu_i, sum_functional_integrals_scalar_covariates_i_t)) - logistic_function(alphas[y_i_t-1] - compute_Z_i_t(mu_i, sum_functional_integrals_scalar_covariates_i_t))
  end
end

function probabilities_ordinal_logistic_i_t_z_available(ordinal_values_low_to_high, alphas, y_i_t, Z_i_t)
  if y_i_t == 1
    return logistic_function(alphas[1] - Z_i_t)
  elseif y_i_t == length(ordinal_values_low_to_high)
    return 1 - logistic_function(alphas[y_i_t-1] - Z_i_t)
  else
    return logistic_function(alphas[y_i_t] - Z_i_t) - logistic_function(alphas[y_i_t-1] - Z_i_t)
  end
end

function probability_ordinal_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i, sum_functional_integrals_scalar_covariates_i_t, link)
  if link == "logit"
    return probabilities_ordinal_logistic_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i, sum_functional_integrals_scalar_covariates_i_t)
  end
end

function get_if_ordinal_logistic(family, link)
  if family == "ordinal" && link == "logit"
    return true
  else
    return false
  end
end

function compute_s_exp_from_alphas(alphas)
  s_exp = Array{Float64}(undef, length(alphas)-1)
  for i in 1:(length(alphas)-1)
    s_exp[i] = exp(alphas[i+1] - alphas[i])
  end
  return s_exp
end


#-------------------------------Evaluate likelihood higher level-----------------
function compute_products_of_probability_per_individual(data_grouped, individual_index,
                                    y_calc, ordinal_values_low_to_high, alphas, mu_i,
                                    sum_functional_integrals_scalar_covariates_i,
                                    link, family)
  prod = 1
  for t_index in 1:nrow(data_grouped[individual_index])
    y_i_t = data_grouped[individual_index][t_index, y_calc]
    prob_i_t = probability_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i,
                          sum_functional_integrals_scalar_covariates_i[t_index],
                          link, family)
    prod = prod * prob_i_t
  end
  return prod
end

function probability_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i, sum_functional_integrals_scalar_covariates_i_t, link, family)
  if get_if_ordinal_logistic(family, link)
    return probability_ordinal_i_t(ordinal_values_low_to_high, alphas, y_i_t, mu_i, sum_functional_integrals_scalar_covariates_i_t, link)
  end
end

function compute_likelihood_contribution_i(kappa, w_h, data_grouped, individual_index, y_calc, ordinal_values_low_to_high, alphas, sigma_mu, sum_functional_integrals_scalar_covariates_i, link, family)
  sum = 0.0
  for h in 1:length(kappa)
    product = compute_products_of_probability_per_individual(data_grouped, individual_index,
                                          y_calc, ordinal_values_low_to_high,
                                          alphas, kappa[h]*sqrt(2)*sigma_mu,
                                          sum_functional_integrals_scalar_covariates_i, link, family)
    sum = sum + w_h[h]*product
  end
  return  sum / sqrt(pi)
end

function compute_negative_log_likelihood_value(kappa, w_h, data_grouped, y_calc,
            ordinal_values_low_to_high, alphas, sigma_mu, t_variable, basis_type_weighting_betas,
            beta_parameter_dict, functional_covariates, functional_data, gammas, link,
            family, name_no_functional_group, gauss_legendre_points, w_gauss_legendre, scalar_covariates,
            length_betas_all, link_sigma_mu_derivative, transformed_sigma_mu)
  sum = 0.0
  for individual_index in 1:length(data_grouped)
    sum = sum -log(compute_likelihood_contribution_i(kappa, w_h, data_grouped, individual_index, y_calc,
                ordinal_values_low_to_high, alphas, sigma_mu, compute_sum_functional_integrals_scalar_covariates_for_one_individual(data_grouped, individual_index,
                                                                t_variable, basis_type_weighting_betas,
                                                                beta_parameter_dict,
                                                                functional_covariates, functional_data, gammas,
                                                                name_no_functional_group, gauss_legendre_points,
                                                                w_gauss_legendre, scalar_covariates, Array{Any}(undef, nrow((data_grouped[individual_index])))),
                link, family))
  end
  return sum
end

function compute_negative_log_likelihood_value_parallel(kappa, w_h, data_grouped, y_calc,
            ordinal_values_low_to_high, alphas, sigma_mu, t_variable, basis_type_weighting_betas,
            beta_parameter_dict, functional_covariates, functional_data, gammas, link,
            family, name_no_functional_group, gauss_legendre_points, w_gauss_legendre, scalar_covariates,
            length_betas_all, link_sigma_mu_derivative, transformed_sigma_mu)


  length_betas = length(collect(Iterators.flatten(values(beta_parameter_dict))))
  length_theta =  length_betas + length(alphas) + length(gammas) + 1

  sum_likelihood = 0.0
  sum_gradients = 0.0#zeros(length_theta)
  for individual_index in 1:length(data_grouped)#Threads.@threads
    sum_functional_integrals_i = compute_sum_functional_integrals_scalar_covariates_for_one_individual(data_grouped, individual_index,
                                                    t_variable, basis_type_weighting_betas,
                                                    beta_parameter_dict,
                                                    functional_covariates, functional_data,
                                                    gammas, name_no_functional_group,
                                                    gauss_legendre_points, w_gauss_legendre, scalar_covariates,
                                                    Array{Any}(undef, nrow((data_grouped[individual_index]))))

    likelihood_contribution_i = compute_likelihood_contribution_i(kappa, w_h, data_grouped,
                                                                  individual_index, y_calc,
                                                                  ordinal_values_low_to_high, alphas,
                                                                  sigma_mu, sum_functional_integrals_i,
                                                                  link, family) #already divided by sqrt pi

    #gradient_contribution_i = compute_gradient_contribution_i(length_theta, kappa, w_h, sigma_mu, sum_functional_integrals_i,
  #                                    individual_index, data_grouped, scalar_covariates, gammas, y_calc,
  #                                    family, link, alphas, ordinal_values_low_to_high,
  #                                    length_betas, t_variable, beta_parameter_dict,
  #                                    functional_covariates,
  #                                    basis_type_weighting_betas, functional_data,
  #                                    gauss_legendre_points, w_gauss_legendre, name_no_functional_group,
  #                                    link_sigma_mu_derivative,
  #                                    likelihood_contribution_i, transformed_sigma_mu)

    sum_likelihood = sum_likelihood - log(likelihood_contribution_i)
    sum_gradients = sum_gradients - 0# gradient_contribution_i

  end


  return sum_likelihood, sum_gradients
end

function compute_ordinal_logistic_derivative_dp_dz_given_y(y_i_t, Z_i_t, alphas,
                                                     ordinal_values_low_to_high)
  if y_i_t == 1
    return derivative_logistic_alpha_Z(alphas[1], Z_i_t)
  elseif y_i_t == length(ordinal_values_low_to_high)
    return  - derivative_logistic_alpha_Z(last(alphas), Z_i_t)
  else
    return  derivative_logistic_alpha_Z(alphas[y_i_t], Z_i_t) - derivative_logistic_alpha_Z(alphas[y_i_t-1], Z_i_t)
  end
end


function compute_derivative_dp_dz_given_y(family, link, y_i_t, Z_i_t, alphas,
                                          ordinal_values_low_to_high)
  if get_if_ordinal_logistic(family, link)
    return compute_ordinal_logistic_derivative_dp_dz_given_y(y_i_t, Z_i_t, alphas,
                                                        ordinal_values_low_to_high)
  end
end

function compute_derivative_dp_dalpha_1_s_given_y_j(j, family, link, y_i_t, Z_i_t, alphas,
                                          ordinal_values_low_to_high)
  if get_if_ordinal_logistic(family, link)
    return compute_ordinal_logistic_derivative_dp_dalpha_1_s_given_y_j(j, alphas, y_i_t,
                                                  Z_i_t, ordinal_values_low_to_high)
  end
end

function compute_ordinal_logistic_derivative_dp_dalpha_1_s_given_y_j(j, alphas, y_i_t, Z_i_t, ordinal_values_low_to_high)
  s_exp = compute_s_exp_from_alphas(alphas)
  if j > y_i_t
    return 0
  end

  if j == 1
    return - compute_ordinal_logistic_derivative_dp_dz_given_y(y_i_t, Z_i_t, alphas,
                                                         ordinal_values_low_to_high)
  elseif j < y_i_t
    return -s_exp[j-1] * compute_ordinal_logistic_derivative_dp_dz_given_y(y_i_t, Z_i_t, alphas,
                                                         ordinal_values_low_to_high)
  elseif j == y_i_t
    return logistic_function(- alphas[j] + Z_i_t )^2 * s_exp[j-1]
  else
    return nothing
  end
end

function compute_derivative_dp_dalpha_1_s_given_y(family, link, y_i_t, Z_i_t, alphas,
                                          ordinal_values_low_to_high)
  derivative_dp_dalpha_1_s_given_y = Array{Float64}(undef, length(alphas))
  for j in 1:length(alphas)
    derivative_dp_dalpha_1_s_given_y[j] = compute_derivative_dp_dalpha_1_s_given_y_j(j, family, link, y_i_t, Z_i_t, alphas,
                                              ordinal_values_low_to_high)
  end
  return derivative_dp_dalpha_1_s_given_y
end

function compute_gradient_contribution_i(length_theta, kappa, w_h, sigma_mu, sum_functional_integrals_i,
                                  individual_index, data_grouped, scalar_covariates, gammas, y_calc,
                                  family, link, alphas, ordinal_values_low_to_high,
                                  length_betas, t_variable, beta_parameter_dict,
                                  functional_covariates,
                                  basis_type_weighting_betas, functional_data,
                                  gauss_legendre_points, w_gauss_legendre, name_no_functional_group,
                                  link_sigma_mu_derivative,
                                  likelihood_contribution_i, transformed_sigma_mu)

  sum_derivative_h = zeros(length_theta)
  for h in 1:length(kappa)
    mu_transformed = kappa[h]*sqrt(2)*sigma_mu

    derivative_sum_over_t = zeros(length_theta)
    sum_scalars_i = Array{Float64}(undef, length(sum_functional_integrals_i))
    for t_index in 1:nrow(data_grouped[individual_index])
      sum_scalars_i[t_index] = compute_inner_vector_product(data_grouped[individual_index][t_index, scalar_covariates], gammas)
      sum_functional_integrals_scalar_covariates_i_t = sum_functional_integrals_i[t_index] + sum_scalars_i[t_index]

      Z_i_t = compute_Z_i_t(mu_transformed, sum_functional_integrals_scalar_covariates_i_t)
      y_i_t = data_grouped[individual_index][t_index, y_calc]

      #compute dp/ds bzw. dp/dalpha given y; use compute_s_exp_from_alphas(alphas)

      derivative_dp_dalpha_1_s_given_y = compute_derivative_dp_dalpha_1_s_given_y(family, link, y_i_t, Z_i_t, alphas,
                                                ordinal_values_low_to_high)


      #dp/dz
      derivative_dp_dz_given_y_j = compute_derivative_dp_dz_given_y(family, link, y_i_t, Z_i_t, alphas,
                                                                ordinal_values_low_to_high)
      derivative_dp_dz_given_y = vcat(repeat([derivative_dp_dz_given_y_j], length_betas+length(gammas)), derivative_dp_dalpha_1_s_given_y,
                                      [derivative_dp_dz_given_y_j])

      #denominator in sum over t
       prob_i_t = probability_i_t(ordinal_values_low_to_high, alphas, y_i_t,
                                        mu_transformed, sum_functional_integrals_scalar_covariates_i_t,
                                        link, family)

      #derivatives of Z with respect to different parameters (betas, gammas, alphas/s, zeta=ln(sigma_mu))
      derivative_dZ_dtheta = compute_dZ_dtheta(t_variable, t_index, individual_index, beta_parameter_dict,
                                    functional_covariates, data_grouped,
                                    basis_type_weighting_betas, functional_data,
                                    gauss_legendre_points, w_gauss_legendre, name_no_functional_group,
                                    scalar_covariates, link_sigma_mu_derivative, transformed_sigma_mu, alphas,
                                    kappa[h])
      #calculate for t

      derivative_sum_over_t = derivative_sum_over_t + (derivative_dZ_dtheta .* derivative_dp_dz_given_y / prob_i_t)

    end #end t

    sum_functional_integrals_scalar_covariates_i = sum_functional_integrals_i + sum_scalars_i
    product_i = compute_products_of_probability_per_individual(data_grouped, individual_index,
                                        y_calc, ordinal_values_low_to_high, alphas, mu_transformed,
                                        sum_functional_integrals_scalar_covariates_i,
                                        link, family)



    sum_derivative_h = sum_derivative_h + (derivative_sum_over_t * product_i * w_h[h])


  end#end h


  sum_derivative_individual_i = sum_derivative_h / likelihood_contribution_i / sqrt(pi)


  return sum_derivative_individual_i #pi to account for pi which we divide in lh contribution
end

#-------------------------Solely derivatives------------------------------------
function compute_dZ_dbeta_derivatives(t_variable, t_index, individual_index, beta_parameter_dict, functional_covariates, data_grouped,
                                  basis_type_weighting_betas, functional_data,
                                  gauss_legendre_points, w_gauss_legendre, name_no_functional_group)

  derivatives_beta = Array{Float64}(undef, length(collect(Iterators.flatten(values(beta_parameter_dict)))))
  index_total = 1
  for i in 1:length(functional_covariates)
    index_functional_covariate = 1
    for j in 1:length(beta_parameter_dict[functional_covariates[i]])

      if data_grouped[individual_index][t_index, functional_covariates[i]] == name_no_functional_group
        derivatives_beta[index_total] = 0
      end
      if functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["min_time"] >= data_grouped[individual_index][t_index, t_variable]
        derivatives_beta[index_total] = 0
      elseif functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["max_time"] + basis_type_weighting_betas[functional_covariates[i]]["memory"] <= data_grouped[individual_index][t_index, t_variable]
        derivatives_beta[index_total] = 0
      else
        derivatives_beta[index_total] = compute_functional_integral_term_derivative_j(index_functional_covariate,
                              data_grouped[individual_index][t_index, t_variable],
                              basis_type_weighting_betas[functional_covariates[i]]["memory"],
                              functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["basis_parameter"],
                              functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["basis"],
                              beta_parameter_dict[functional_covariates[i]],
                              basis_type_weighting_betas[functional_covariates[i]]["basis"],
                              functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["min_time"],
                              functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["max_time"],
                              functional_data[functional_covariates[i]][data_grouped[individual_index][t_index, functional_covariates[i]]]["link_function"],
                              basis_type_weighting_betas[functional_covariates[i]]["link_function_derivative"],
                              gauss_legendre_points, w_gauss_legendre)
      end
      index_total = index_total + 1
      index_functional_covariate = index_functional_covariate + 1
    end
  end
  return derivatives_beta
end

function compute_dZ_dgamma_derivatives(scalar_covariates, data_grouped, individual_index,
                                   t_index)
  gamma_derivatives = Array{Float64}(undef, length(scalar_covariates))
  for i in 1:length(scalar_covariates)
    gamma_derivatives[i] = data_grouped[individual_index][t_index, scalar_covariates[i]]
  end
  return gamma_derivatives
end

function compute_dZ_dtheta(t_variable, t_index, individual_index, beta_parameter_dict,
                           functional_covariates, data_grouped,
                           basis_type_weighting_betas, functional_data,
                           gauss_legendre_points, w_gauss_legendre, name_no_functional_group,
                           scalar_covariates, link_sigma_mu_derivative, transformed_sigma_mu,
                           alphas, kappa)

  dZ_dbeta_derivatives = compute_dZ_dbeta_derivatives(t_variable, t_index, individual_index, beta_parameter_dict,
                                              functional_covariates, data_grouped,
                                              basis_type_weighting_betas, functional_data,
                                              gauss_legendre_points, w_gauss_legendre, name_no_functional_group)

  dZ_dgamma_derivatives = compute_dZ_dgamma_derivatives(scalar_covariates, data_grouped, individual_index,
                                     t_index)

  dZ_dzeta_derivative = kappa*sqrt(2)*link_sigma_mu_derivative(transformed_sigma_mu)
  dZ_dalpha_derivatives = ones(length(alphas)) #only helper

  return vcat(dZ_dbeta_derivatives, dZ_dgamma_derivatives, dZ_dalpha_derivatives,
              [dZ_dzeta_derivative])
end

#------------------------Penalty related----------------------------------------
function compute_penalty_g(t, basis_k, beta_k)
  if check_if_basis_is_BSpline(basis_k)
    return compute_sum_offset_array(bsplines(basis_k, t, Derivative(2)), beta_k)^2
  end
end

function compute_penalty_integral_k(basis_k, beta_k, memory_k, gauss_legendre_points_penalty, w_gauss_legendre_penalty)
  sum = 0.0
  for i in 1:length(gauss_legendre_points_penalty)
    sum = sum + w_gauss_legendre_penalty[i] * compute_penalty_g(memory_k/2*(gauss_legendre_points_penalty[i]+1), basis_k, beta_k)
  end
  return sum
end

function compute_penalty_sum_integrals(functional_covariates, basis_type_weighting_betas, beta_parameter_dict,
                                      gauss_legendre_penalty_points)
  sum = 0.0
  for i in 1:length(functional_covariates)
    sum = sum + compute_penalty_integral_k(basis_type_weighting_betas[functional_covariates[i]]["basis"],
                beta_parameter_dict[functional_covariates[i]], basis_type_weighting_betas[functional_covariates[i]]["memory"],
                gauss_legendre_penalty_points[functional_covariates[i]]["points"], gauss_legendre_penalty_points[functional_covariates[i]]["weights"])
  end
  return sum
end
#-------------------------Objective---------------------------------------------
function compute_objective_function_value(betas, gammas, alphas, transformed_sigma_mu,
                                          lambda,
                                          kappa, w_h, data_grouped, y_calc, length_betas_all,
                                          ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                                          functional_covariates, functional_data, link, family,
                                          name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                          link_sigma_mu::Function, link_sigma_mu_derivative,
                                          parallel, scalar_covariates, gauss_legendre_penalty_points
                                          )

  beta_parameter_dict = create_beta_dict(betas, length_betas_all)
  if parallel
    function_value, gradient = compute_negative_log_likelihood_value_parallel(kappa, w_h, data_grouped, y_calc,
                ordinal_values_low_to_high, alphas, link_sigma_mu(transformed_sigma_mu), t_variable, basis_type_weighting_betas,
                beta_parameter_dict, functional_covariates, functional_data, gammas, link, family,
                name_no_functional_group, gauss_legendre_points, w_gauss_legendre, scalar_covariates,
                length_betas_all, link_sigma_mu_derivative, transformed_sigma_mu)

    return  function_value + lambda * compute_penalty_sum_integrals(functional_covariates,
                             basis_type_weighting_betas, beta_parameter_dict,
                             gauss_legendre_penalty_points), gradient, function_value

  else

    return compute_negative_log_likelihood_value(kappa, w_h, data_grouped, y_calc,
                ordinal_values_low_to_high, alphas, link_sigma_mu(transformed_sigma_mu), t_variable, basis_type_weighting_betas,
                beta_parameter_dict, functional_covariates, functional_data, gammas, link, family,
                name_no_functional_group, gauss_legendre_points, w_gauss_legendre, scalar_covariates, length_betas_all, link_sigma_mu_derivative, transformed_sigma_mu) +
                lambda * compute_penalty_sum_integrals(functional_covariates, basis_type_weighting_betas, beta_parameter_dict,
                                                       gauss_legendre_penalty_points)
  end
end

function get_wrapper_compute_objective_function_value(theta, length_alphas, scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            link_sigma_mu, link_sigma_mu_derivative,
                            lambda, parallel, gauss_legendre_penalty_points)

  length_betas = sum(values(length_betas_all))
  length_gammas = length(scalar_covariates)

  if get_if_ordinal_logistic(family, link)
    alpha_1_and_s = theta[(length_betas+length_gammas+1):(length_betas+length_gammas+length_alphas)]
    return compute_objective_function_value(theta[1:length_betas],
                                          theta[(length_betas+1):(length_betas+length_gammas)],
                                          get_alpha_from_s_and_alpha_1(alpha_1_and_s[2:length(alpha_1_and_s)], alpha_1_and_s[1]),
                                          last(theta),
                                          lambda,
                                          kappa, w_h, data_grouped, y_calc, length_betas_all,
                                          ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                                          functional_covariates, functional_data, link, family,
                                          name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                          link_sigma_mu, link_sigma_mu_derivative,
                                          parallel, scalar_covariates, gauss_legendre_penalty_points
                                          )
  else
    return compute_objective_function_value(theta[1:length_betas],
                                          theta[(length_betas+1):(length_betas+length_gammas)],
                                          theta[(length_betas+length_gammas+1):(length_betas+length_gammas+length_alphas)],
                                          last(theta),
                                          lambda,
                                          kappa, w_h, data_grouped, y_calc, length_betas_all,
                                          ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                                          functional_covariates, functional_data, link, family,
                                          name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                          link_sigma_mu, link_sigma_mu_derivative,
                                          lambda, parallel, scalar_covariates, gauss_legendre_penalty_points
                                          )
  end
end

function get_alpha_from_s_and_alpha_1(s_vector, alpha_1)
    new_alpha = [alpha_1]
    for i in 1:length(s_vector)
      push!(new_alpha, alpha_1 + sum(identity_function.(s_vector[1:i]))) #identity if bounded optimization
    end
    return new_alpha
end

function get_optimization_results(starting_values, length_alphas, scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            gauss_legendre_penalty_points,
                            link_sigma_mu, link_sigma_mu_derivative,
                            lambda, parallel, optimizer=LBFGS())

  function f(theta, length_alphas, scalar_covariates, #f only needed to not use gradient -> otherwise dircetly get_wrapper_compute_objective_function_value
                              kappa, w_h, data_grouped, y_calc, length_betas_all,
                              ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                              functional_covariates, functional_data, link, family,
                              name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                              link_sigma_mu, link_sigma_mu_derivative,
                              lambda, parallel, gauss_legendre_penalty_points)
    return get_wrapper_compute_objective_function_value(theta, length_alphas, scalar_covariates,
                                kappa, w_h, data_grouped, y_calc, length_betas_all,
                                ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                                functional_covariates, functional_data, link, family,
                                name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                link_sigma_mu, link_sigma_mu_derivative,
                                lambda, parallel, gauss_legendre_penalty_points)[1]
  end


  lower = vcat(repeat([-Inf], sum(collect(values(length_betas_all) ))), repeat([-Inf],  length(scalar_covariates) + 1), repeat([10^(-30)], length_alphas-1), [0])
  upper = vcat(repeat([Inf], sum(collect(values(length_betas_all) ))),      repeat([Inf], length(starting_values) - sum(collect(values(length_betas_all) )) )   )


  fn = OnceDifferentiable(theta -> f(theta, length_alphas, scalar_covariates,
                              kappa, w_h, data_grouped, y_calc, length_betas_all,
                              ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                              functional_covariates, functional_data, link, family,
                              name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                              link_sigma_mu, link_sigma_mu_derivative,
                              lambda, parallel, gauss_legendre_penalty_points), starting_values,
                              autodiff=:forward)

  fit = optimize(fn, #g!,
                                      lower, upper, starting_values,
                                      optimizer, Optim.Options(#show_trace = true, #show_every = 1,
                                      iterations=1000, g_tol = 10^(-6), f_tol = 10^(-20)))#,
                                      #store_trace=true, extended_trace=true, iterations = 3000))
  parameter = Optim.minimizer(fit)
  likelihood_value = get_wrapper_compute_objective_function_value(parameter, length_alphas, scalar_covariates,
                                                          kappa, w_h, data_grouped, y_calc, length_betas_all,
                                                          ordinal_values_low_to_high, t_variable, basis_type_weighting_betas,
                                                          functional_covariates, functional_data, link, family,
                                                          name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                                          link_sigma_mu, link_sigma_mu_derivative,
                                                          lambda, parallel, gauss_legendre_penalty_points)[3]
  output = Dict("parameter" => parameter,
                "log_likelihood" => -likelihood_value,
                "fit" => fit)

  return output
end


function fun_glmm(formula, family, link, data, functional_data, beta_bases, starting_values,
                  i_variable = "id", t_variable = "time", n_gauss_hermite_quadrature_points_random_effect=20,
                  n_gauss_legendre_points_functional_covariate=10, lambda=10,
                  name_no_functional_group = "None", parallel_likelihood_evaluation=false,
                  link_sigma_mu = exponential_function, link_sigma_mu_derivative=exponential_function,
                  ordinal_values_low_to_high=nothing,
                  optimizer=LBFGS(), bootstrap=nothing)

  #extract formula names to names of variables
  y_name = extract_dependent(formula)
  functional_covariates, scalar_covariates = extract_X_names(formula)

  #model pre-processing if ordinal_logistic
  if get_if_ordinal_logistic(family, link)
    data[!, "y_cat"] = create_numerical_ordinal_values(data[!, y_name], ordinal_values_low_to_high)
  end
  #get name of y_variable (changes if ordinal logistic)
  y_calc = get_y_calc(family, y_name)

  #get length of covariate parameters to optimize over
  length_betas_all = compute_beta_basis_lengths(functional_covariates, beta_bases)
  length_alphas = get_length_alphas(family, link, ordinal_values_low_to_high)

  #get all needed quadrature points
  kappa, w_h = gausshermite(n_gauss_hermite_quadrature_points_random_effect)
  gauss_legendre_points, w_gauss_legendre = gausslegendre(n_gauss_legendre_points_functional_covariate)
  gauss_legendre_penalty_points = compute_gauss_legendre_penalty_points(beta_bases)

  data_grouped = groupby(data, i_variable) #group by individuals


  #obtain results of the penalized mll function
  output = get_optimization_results(starting_values, length_alphas, scalar_covariates,
                              kappa, w_h, data_grouped, y_calc, length_betas_all,
                              ordinal_values_low_to_high, t_variable, beta_bases,
                              functional_covariates, functional_data, link, family,
                              name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                              gauss_legendre_penalty_points,
                              link_sigma_mu, link_sigma_mu_derivative, lambda,
                              parallel_likelihood_evaluation, optimizer)

  output["aic"] = compute_aic(length(output["parameter"]), output["log_likelihood"])
  output["bic"] = compute_bic(length(output["parameter"]), nrow(data), output["log_likelihood"])

  if !isnothing(bootstrap) #idea of grouped influence boostrap?
    output["bootstrap"] = run_bootstrap(Array{Float64}(undef, bootstrap, length(starting_values)), bootstrap,
                                starting_values, length_alphas,
                                scalar_covariates,
                                kappa, w_h, data_grouped, y_calc, length_betas_all,
                                ordinal_values_low_to_high, t_variable, beta_bases,
                                functional_covariates, functional_data, link, family,
                                name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                gauss_legendre_penalty_points,
                                link_sigma_mu, link_sigma_mu_derivative, lambda,
                                parallel_likelihood_evaluation, optimizer, i_variable)
   end

   return output

end

#--------------------------Model evaluation-------------------------------------
function compute_bic(n_parameter, n_observations, log_likelihood)
  return n_parameter*log(n_observations) - 2*log_likelihood
end

function compute_aic(n_parameter, log_likelihood)
  return 2*n_parameter - 2 * log_likelihood
end

#--------------------------------Bootstrap--------------------------------------
function run_bootstrap(store, bootstrap,
                            starting_values, length_alphas,
                            scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, beta_bases,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            gauss_legendre_penalty_points,
                            link_sigma_mu, link_sigma_mu_derivative, lambda,
                            parallel_likelihood_evaluation, optimizer, i_variable)

  println("Bootstrap progress")
  p = Progress(bootstrap) #just for progressbar
  update!(p, 0) #just for progressbar
  jj = Threads.Atomic{Int}(0) #just for progressbar
  l = Threads.SpinLock() #just for progressbar

  Threads.@threads for b in 1:bootstrap
    store[b,:] = run_bootstrap_b(1, length(data_grouped), length(data_grouped), true,
                                starting_values, length_alphas,
                                scalar_covariates,
                                kappa, w_h, data_grouped, y_calc, length_betas_all,
                                ordinal_values_low_to_high, t_variable, beta_bases,
                                functional_covariates, functional_data, link, family,
                                name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                                gauss_legendre_penalty_points,
                                link_sigma_mu, link_sigma_mu_derivative, lambda,
                                parallel_likelihood_evaluation, optimizer, i_variable)["parameter"]

    Threads.atomic_add!(jj, 1) #just for progressbar
    Threads.lock(l) #just for progressbar
    update!(p, jj[]) #just for progressbar
    Threads.unlock(l)  #just for progressbar
  end

  return store
end


function run_bootstrap_b(mini, maxi, n, replacement, starting_values, length_alphas,
                            scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, beta_bases,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            gauss_legendre_penalty_points,
                            link_sigma_mu, link_sigma_mu_derivative, lambda,
                            parallel_likelihood_evaluation, optimizer, i_variable)

  return get_optimization_results(starting_values, length_alphas, scalar_covariates,
                              kappa, w_h, draw_bootstrap_data_set(data_grouped, mini, maxi, n, replacement, i_variable),
                              y_calc, length_betas_all,
                              ordinal_values_low_to_high, t_variable, beta_bases,
                              functional_covariates, functional_data, link, family,
                              name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                              gauss_legendre_penalty_points,
                              link_sigma_mu, link_sigma_mu_derivative, lambda,
                              parallel_likelihood_evaluation, optimizer)
end


function draw_bootstrap_data_set(data_grouped, mini, maxi, n, replacement, i_variable)
  return groupby(reduce(vcat, [data_grouped[i] for i in sample_integers(mini, maxi, n, replacement)]), i_variable)
end


function sample_integers(mini, maxi, n, replacement)
  return sample([mini:maxi;], n, replace=replacement)
end


#-------------------------------Data Set simulation-----------------------------
function simulate_functional_groups_and_times(functional_covariates, min_obs_per_individual,
                                max_obs_per_individual, min_time, max_time, id_i)

  name_functional_covariates = collect(keys(functional_covariates))
  number_obs = sample(min_obs_per_individual:max_obs_per_individual, 1)[1]
  time_points = sort(rand(Uniform(min_time, max_time), number_obs))
  df_i = DataFrame(time = time_points, id = repeat([id_i], number_obs))
  for i in 1:length(name_functional_covariates)
     df_i[!, name_functional_covariates[i]] = repeat(sample(collect(keys(functional_covariates[name_functional_covariates[i]])), 1), number_obs)
  end
  return df_i
end

function compute_scalar_covariates_and_groups(number_individuals, functional_covariates, min_obs_per_individual,
                                  max_obs_per_individual, min_time, max_time, variance_scalar_covariates,
                                  number_scalar_covariates, sigma_mu)
  df = DataFrame()
  for i in 1:number_individuals
    df_1 = simulate_functional_groups_and_times(functional_covariates, min_obs_per_individual,
                                      max_obs_per_individual, min_time, max_time, "id_$i")
    df_1[!, "mu_i"] = repeat(rand(Normal(0, sigma_mu), 1), nrow(df_1))
    df = [df; df_1]
  end
  scalar_covariates_values = DataFrame(reshape(rand(Normal(0, variance_scalar_covariates), nrow(df)*number_scalar_covariates),
                                (nrow(df), number_scalar_covariates)), :auto )
  df_scalars_covariates = hcat(df, scalar_covariates_values)

  #df_scalars_covariates[!, "mu_i"] = rand(Normal(0, sigma_mu), nrow(df))

  return df_scalars_covariates
end

function compute_functional_covariate(grid_t)
  w =  Array{Float64}(undef, last(grid_t))
  w[1] = rand(Normal(0, 1), 1)[1]
  for i in 2:last(grid_t)
    w[i] = w[i-1] + 0.5*sin(2*pi / last(grid_t) * 7 * i ) + 0.5*cos(2*pi / last(grid_t) * 4 * i ) + rand(Normal(0, 1), 1)[1]
  end

  w_scale = (w .- minimum(w)) ./ (maximum(w) - minimum(w) )
  return w_scale
end

function calculate_functional_covariates(functional_covariates)
  names_functional_covariates = collect(keys(functional_covariates))
  for i in 1:length(names_functional_covariates)
    subgroups = collect(keys(functional_covariates[names_functional_covariates[i]]))
    for j in 1:length(subgroups)
      if !haskey(functional_covariates[names_functional_covariates[i]][subgroups[j]], "data_frame")
        grid_t = functional_covariates[names_functional_covariates[i]][subgroups[j]]["min_time"]:functional_covariates[names_functional_covariates[i]][subgroups[j]]["max_time"]
        functional_covariates[names_functional_covariates[i]][subgroups[j]]["data_fram"] = compute_functional_covariate(grid_t)
      end
    end
  end
  return functional_covariates
end

function compute_alphas_test_data(weights_y_data, Z)
  alphas = Array{Float64}(undef, length(weights_y_data)-1)
  for i in 1:(length(weights_y_data)-1)
    alphas[i] = percentile(Z, 100*sum(weights_y_data[1:i]))
  end
  return alphas
end

function simulate_y(data, ordinal_values_low_to_high, weights_y_data, alphas)
  y_sim = Array{String}(undef, nrow(data))
  for i in 1:nrow(data)
    probabilities = [probabilities_ordinal_logistic_i_t_z_available(ordinal_values_low_to_high,
                                                  alphas, j, data[i, "Z"]) for j in 1:length(weights_y_data)]
    y_sim[i] = sample(ordinal_values_low_to_high, Weights(probabilities))
  end
  return y_sim
end

function simulate_data(family, link, lower_bound_gamma, upper_bound_gamma,
                        number_individuals, min_time,max_time, number_gauss_legendre_points,
                        number_scalar_covariates, max_obs_per_individual, min_obs_per_individual,
                        variance_scalar_covariates,  bases_betas, functional_covariates,
                        ordinal_values_low_to_high=nothing,
                        weights_y_data=nothing)
  scalar_covariates = ["x$v" for v in 1:number_scalar_covariates]
  gauss_legendre_points, w_gauss_legendre = gausslegendre(number_gauss_legendre_points)
  length_betas = compute_beta_basis_lengths(collect(keys(functional_covariates)), bases_betas)
  betas = get_dummy_beta_parameters_development(length_betas)

  gammas = rand(Uniform(lower_bound_gamma, upper_bound_gamma), number_scalar_covariates)
  beta_parameter_dict = create_beta_dict(betas, length_betas)

  functional_covariates_calculated = calculate_functional_covariates(functional_covariates)

  df = compute_scalar_covariates_and_groups(number_individuals, functional_covariates, min_obs_per_individual,
                                       max_obs_per_individual, min_time, max_time, variance_scalar_covariates,
                                       number_scalar_covariates, sigma_mu)
  data_grouped = groupby(df, "id")

  for i in eachindex(data_grouped)
    sum_functional_plus_scalar_covariates = compute_sum_functional_integrals_scalar_covariates_for_one_individual(data_grouped, i,
                                                  "time", bases_betas,
                                                  beta_parameter_dict,
                                                  collect(keys(functional_covariates)),
                                                  functional_covariates_calculated,
                                                  gammas, "None",
                                                  gauss_legendre_points, w_gauss_legendre,
                                                  scalar_covariates, Array{Any}(undef, nrow((data_grouped[i]))))
    Z_i = compute_Z_i_t.(data_grouped[i][!,"mu_i"], sum_functional_plus_scalar_covariates)
    data_grouped[i][!, "Z"] = Z_i
  end

  data = DataFrame(data_grouped)

  if get_if_ordinal_logistic(family, link)
    alphas = compute_alphas_test_data(weights_y_data, data[!, "Z"])
    data[!,"y"] = simulate_y(data, ordinal_values_low_to_high, weights_y_data, alphas)
  end

  return Dict("data" => data, "alphas" => alphas, "gammas" => gammas, "betas" => beta_parameter_dict,
              "functional_data" => functional_covariates_calculated)
end

function compute_s_from_differences(alphas)
  d = Array{Float64}(undef, length(alphas))
  d[1] = alphas[1]
  for i in 2:length(alphas)
    d[i] = alphas[i] - alphas[i-1] #no log differences
  end
  return d
end

#------------------------------Visualization (improve)--------------------------
function plot_results(bases_betas,
                      estimated_parameters, true_values, start_values, bootstrap_parameters,
                      level_confidence =0.05,
                     link_function=identity_function)
  PP = []

  functional_covariates = collect(keys(bases_betas))
  length_betas = Array{Float64}(undef, length(functional_covariates))
  for i in 1:length(functional_covariates)

    name_fc = functional_covariates[i]
    basis_ = bases_betas[name_fc]["basis"]
    length_betas[i] = length(basis_)
    B = basismatrix(basis_, ( 1:bases_betas[functional_covariates[i]]["memory"]))
    index_start = Integer((sum(length_betas[1:(i-1)])))
    beta_est = B * estimated_parameters[(index_start+1):(index_start+length(basis_))]
    beta_true = B * true_values[(index_start+1):(index_start+length(basis_))]
    beta_start = B * start_values[(index_start+1):(index_start+length(basis_))]

    bootstrap_results = B*bootstrap_parameters[:, (index_start+1):(index_start+length(basis_))]'
    row_means = mean(bootstrap_results, dims=2)
    row_sd = (var(bootstrap_results, dims=2)).^0.5
    k = quantile(Normal(0.0, 1.0),level_confidence/2)
    lower = (k .* row_sd)

    push!(PP, plot(hcat(beta_est, beta_true, beta_start),
         ribbon=hcat(lower, zeros(length(beta_true)),zeros(length(beta_start)) ),
         fc=:steelblue, fa=0.2,
         linewidth = 1.5,
          xlabel = "Time", ylabel="Beta function $name_fc",
          labels=["Estimated" "True function" "Start function"]) )
  end
  plot(PP...; size = default(:size) .* (1, length(functional_covariates)),
   layout = (length(functional_covariates), 1), left_margin = 5Plots.mm)
end

function plot_bases_functions(basis)
  bp = breakpoints(basis)
  B = basismatrix(basis, [bp[1]:0.1:last(bp);])
  plot([bp[1]:0.1:last(bp);], B, xlabel = "Time", ylabel="Basis function")
end

function plot_results_confidence_intervals(bases_betas,
                      estimated_parameters, true_values, start_values,
                       link_function=identity_function)

  functional_covariates = collect(keys(bases_betas))
  length_betas = Array{Float64}(undef, length(functional_covariates))

  name_fc = functional_covariates[1]
  basis_ = bases_betas[name_fc]["basis"]

  B = basismatrix(basis_, ( 1:bases_betas[functional_covariates[1]]["memory"]))
  index_start = 1
  beta_est = B * estimated_parameters[:, (index_start+1):(index_start+length(basis_))]'
  beta_true = B * true_values[(index_start+1):(index_start+length(basis_))]
  beta_start = B * start_values[(index_start+1):(index_start+length(basis_))]

  plot(hcat(beta_est, beta_true),
          xlabel = "Time", ylabel="Beta function $name_fc", color="steelblue", alpha = 0.7,
      )

end




#--------------------------------Test algorithms--------------------------------

#needed only for simulations
const sigma_mu = 1.0
const lower_bound_gamma = -0.5
const upper_bound_gamma = 0.5
const number_individuals = 5000
const min_time = -600
const max_time = 811
const weights_y_data = [1/3, 1/3, 1/3]
const max_obs_per_individual = 4
const min_obs_per_individual = 4
const variance_scalar_covariates = 1
const number_scalar_covariates = 2


#needed for simulation and regression
const family = "ordinal"
const link = "logit"
const number_gauss_legendre_points = 10
const basis = BSplineBasis(3, [0:15:30;])

const bases_betas = Dict("containment_1" => Dict("basis" => basis, #specify types of contiunous beta coefficients
                                              "link_function" => identity_function,#logistic_function,
                                              "link_function_derivative" => constant_one,
                                              "memory" => 30,
                                              "gauss_legendre_points_penalty" => 5))#,
                                  #"containment_2" => Dict("basis" => BSplineBasis(3, [0:3:9;]),
                                  #            "link_function" => logistic_function,#logistic_function,
                                  #            "link_function_derivative" => derivative_logistic,
                                  #            "memory" => 9,
                                  #            "gauss_legendre_points_penalty" => 7))
const functional_covs = delete!(add_min_max_time(load_functional_test_data_dict("/home/manuel/Documents/fun_glm/test_data/",
                 ["Spain"], 4, [1:18:811;],
                 log, 2, "time", "containment_index"), "time"), "containment_2")
const ordinal_order = ["ordinal_1", "ordinal_2", "ordinal_3"]
functional_covs["containment_1"]["Spain_1"]["data_frame"][!, "containment_index"] = functional_covs["containment_1"]["Spain_1"]["data_frame"][!, "containment_index"] * 100

#simulate data
Random.seed!(12)
simulated_data_dict = simulate_data(family, link, lower_bound_gamma, upper_bound_gamma,
                              number_individuals, min_time, max_time, number_gauss_legendre_points,
                              number_scalar_covariates, max_obs_per_individual, min_obs_per_individual,
                              variance_scalar_covariates,  bases_betas, functional_covs,
                              ordinal_order,
                              weights_y_data)

#plot functional covariates and bases
plot(functional_covs["containment_1"]["Spain_1"]["data_frame"][!, "containment_index"])
plot_bases_functions(basis)

#check distribution of ys
freqtable(simulated_data_dict["data"][!, "y"]) ./ nrow(simulated_data_dict["data"])

#only for regression
const formula = "y ~ x1 + x2 + f(containment_1)"
const true_values = vcat(simulated_data_dict["betas"]["containment_1"],
                          #simulated_data_dict["betas"]["containment_2"],
                          simulated_data_dict["gammas"],
                          simulated_data_dict["alphas"], [sigma_mu])
const start_values_noise =  zeros(length(true_values)) .+ 0.01 #true_values + rand(Uniform(-1, 1), length(true_values))
const time_var = "time"

#TODO
#inverse hessian standard errors
#start values to improve speed of convergence;
fit = fun_glmm(formula, family, #family -> for ordinal not really true cause we do not model E(Y|X,W)
                link, #link function of fun-glmm
                simulated_data_dict["data"], #data on individuals (ordinary panel set)
                functional_covs, #functional covariates
                bases_betas, #beta basis definition
                start_values_noise, #starting values; find nice start
                "id", #name of individual column
                time_var, #name of time column
                10, #number of gauss hermite qudrature points
                number_gauss_legendre_points, #number of gauss legendre points
                0, #lambda penalty / penalty is currently in gradient missing
                "None", #name of no
                true, #parallel "likelihood evaluation -> remove option and only do bootstrap parallel
                identity_function,#exponential_function, #link of mu -> estimate ln(mu)
                constant_one,#exponential_function, #derivative of mu link
                ordinal_order, #only needed for ordinal data
                Fminbox(BFGS()), #type of optimizer
                10 #number of bootstrap draws
                )

estimated_parameters = fit["parameter"]
plot_results(bases_betas, estimated_parameters, true_values,
 start_values_noise, fit["bootstrap"], 0.05, identity_function)
savefig("/home/manuel/Documents/fun_glm/beta_est_many_ind.png")


a = start_values_noise - estimated_parameters
b = true_values - estimated_parameters





#-------------------Debugging-------------------------------------------
y_name = extract_dependent(formula)
functional_covariates, scalar_covariates = extract_X_names(formula)
data = simulated_data_dict["data"]
ordinal_values_low_to_high = ordinal_order
beta_bases = bases_betas
i_variable = "id"
t_variable = "time"
functional_data = functional_covs

#model pre-processing if ordinal_logistic
if get_if_ordinal_logistic(family, link)
  data[!, "y_cat"] = create_numerical_ordinal_values(data[!, y_name], ordinal_values_low_to_high)
end
#get name of y_variable (changes if ordinal logistic)
y_calc = get_y_calc(family, y_name)

#get length of covariate parameters to optimize over
length_betas_all = compute_beta_basis_lengths(functional_covariates, beta_bases)
length_alphas = get_length_alphas(family, link, ordinal_values_low_to_high)
n_gauss_hermite_quadrature_points_random_effect = 10
n_gauss_legendre_points_functional_covariate = 10
#get all needed quadrature points
kappa, w_h = gausshermite(n_gauss_hermite_quadrature_points_random_effect)
gauss_legendre_points, w_gauss_legendre = gausslegendre(n_gauss_legendre_points_functional_covariate)
gauss_legendre_penalty_points = compute_gauss_legendre_penalty_points(beta_bases)

data_grouped = groupby(data, i_variable) #group by individuals

theta = true_values


link_sigma_mu = identity_function
link_sigma_mu_derivative = constant_one
lambda = 0
name_no_functional_group = "None"
an = theta -> get_wrapper_compute_objective_function_value(theta, length_alphas, scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, bases_betas,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            link_sigma_mu, link_sigma_mu_derivative,
                            lambda, true, gauss_legendre_penalty_points)[1]

g = x -> ForwardDiff.gradient(an, start_values_noise);

g(start_values_noise)

theta_mod = copy(theta)
k =2
theta_mod[k] = theta[k] + 10
bn= theta -> get_wrapper_compute_objective_function_value(theta_mod, length_alphas, scalar_covariates,
                            kappa, w_h, data_grouped, y_calc, length_betas_all,
                            ordinal_values_low_to_high, t_variable, bases_betas,
                            functional_covariates, functional_data, link, family,
                            name_no_functional_group, gauss_legendre_points, w_gauss_legendre,
                            link_sigma_mu, link_sigma_mu_derivative,
                            lambda, true, gauss_legendre_penalty_points)



length_betas = sum(values(length_betas_all))
length_gammas = length(scalar_covariates)
alpha_1_and_s = theta[(length_betas+length_gammas+1):(length_betas+length_gammas+length_alphas)]
betas = theta[1:length_betas]
gammas = theta[(length_betas+1):(length_betas+length_gammas)]
alphas = get_alpha_from_s_and_alpha_1(alpha_1_and_s[2:length(alpha_1_and_s)], alpha_1_and_s[1])
zeta = last(theta)
transformed_sigma_mu = link_sigma_mu(zeta)
beta_parameter_dict = create_beta_dict(betas, length_betas_all)
basis_type_weighting_betas = bases_betas

compute_negative_log_likelihood_value_parallel(kappa, w_h, data_grouped, y_calc,
            ordinal_values_low_to_high, alphas, link_sigma_mu(transformed_sigma_mu), t_variable, basis_type_weighting_betas,
            beta_parameter_dict, functional_covariates, functional_data, gammas, link, family,
            name_no_functional_group, gauss_legendre_points, w_gauss_legendre, scalar_covariates,
            length_betas_all, link_sigma_mu_derivative, transformed_sigma_mu)


Threads.@threads for i in 1:40
  println(rand(Uniform(-5, 5), 1))
end
