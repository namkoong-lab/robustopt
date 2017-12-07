# SimpleProjections.jl
#
# A small module for solving a distributionally robust optimization on
# the chi-square divergence ball on the simplex. The bisection
# algorithm takes a desired (relative) solution tolerance as a
# parameter, and is extremely quick for reasonable values of n and
# tolerance.
#
# Uses careful book-keeping and a number of sorted vectors to actually
# perform the projections, which require time
#
# O(n * log(n) + (log(n) * log(1 / epsilon)))
#
# where the first term is a sort, the second is a careful binary search.

module SProj

function RobustExpectation(z::Vector{Float64}, rho::Float64;
                           sol_tolerance::Float64 = 1e-13)
  return dot(z, MinimizeLinearSquareDivergence(z, rho; sol_tolerance = sol_tolerance));
end

# x = MinimizeLinearSquareDivergence(z, rho)
#
# Sets x to be the minimizer of
#
# minimize  sum(x .* z)
#   s.t.    (1/2n) * sum((x * n - 1)^2) <= rho
#           x >= 0, sum(x) == 1.
#
# Does this by performing a binary search on the partially dualized form
#
# maximize  inf_{x >= 0, sum(x) == 1}
#             lambda * ((1/2) * norm(x - 1/n)^2 - rho / n) + x' * z,
#
# by performing a binary search over lambda. With appropriate
# book-keeping, each successive step of the binary search requires
# time O(log(n)).
function MinimizeLinearSquareDivergence(z::Vector{Float64}, rho::Float64;
                                        sol_tolerance::Float64 = 1e-13)
  z_sort = sort(z - mean(z)); # just normalization
  nn = length(z_sort);
  if all(z_sort .== 0)
    return (1/nn) * ones(nn);
  end
  lambda_min = 0;
  lambda_init_max = max(maximum(abs(z_sort)) * nn,
                        sqrt(nn / (2 * rho)) * norm(z_sort));
  lambda_max = lambda_init_max;
  sum_vec = cumsum(z_sort);
  sum_squared_vec = cumsum(z_sort.^2);
  z_norm_squared = sum(z_sort.^2);

  while (abs(lambda_max - lambda_min) > sol_tolerance * lambda_init_max)
    lambda = (lambda_max + lambda_min) / 2;
    (eta, ind) = FindShiftVectorOntoSimplex(z_sort, lambda, sum_vec);

    # Now compute the derivative w.r.t. lambda, which is
    #
    # norm(x - (1/n))^2 - rho / n.
    #
    # We can do this in closed form using our book-keeping
    square_sum = (sum_squared_vec[ind] / lambda^2 + ind * eta^2
                  + 2 * sum_vec[ind] * eta / lambda + (nn - ind) / nn^2);
    if (square_sum / 2 - rho / nn > 0)
      lambda_min = lambda;
    else
      lambda_max = lambda;
    end
  end
  lambda = (lambda_max + lambda_min) / 2;
  (eta, ind) = FindShiftVectorOntoSimplex(z_sort, lambda, sum_vec);
  x = max(ones(nn) / nn - (z - mean(z)) / lambda - eta, 0);
  return x;
end

# (eta, ind) = FindShiftVectorOntoSimplex(z::Vector{Float64}, lambda::Float64,
#                                         s::Vector{Float64})
#
# Let v = (1/n) - z / lambda, where z and s are n-vectors and
# lambda > 0.  Finds the eta such that if
#
#   x_i = (1/n - z / lambda - eta)_+
#
# then sum_i x_i = 1. The vector z must be in ascending sorted order,
# so that z[1] <= z[2] <= ... <= z[n]. The vector s is the cumulative
# sum of the z vector, so that s[i] = sum_{j = 1}^i z[j].
#
# This is equivalent to finding the projection of the vector
#
#   v = 1/n - z / lambda
#
# onto the probability simplex. To do so, we find the index i such that
#
# sum_{j = 1}^i (v[j] - v[i]) < 1
#
# and
#
# sum_{j = 1}^{i + 1} (v[j] - v[i + 1]) >= 1.
#
# This method is explained in Figure 1 of Duchi et al., Efficient
# Projections onto the L1-Ball for Learning in High Dimensions, ICML
# 2008.
#
# Method runs in O(log n) time, and returns the pair (eta, ind) where
# ind satisfies the two preceding inequalities when i = ind.
function FindShiftVectorOntoSimplex(z::Vector{Float64}, lambda::Float64,
                                    sumvec::Vector{Float64})
  nn = length(z);
  low_ind = 1;
  high_ind = nn;
  if (((1/nn) - z[nn] / lambda + (1/nn) * sumvec[end] / lambda) > 0)
    return (-(1/nn) * sumvec[end] / lambda, nn);
  end
  
  while (low_ind != high_ind)
    ii = floor(Int64, (high_ind + low_ind) / 2);
    left_sum = (ii * z[ii] - sumvec[ii]) / lambda;
    right_sum = ((ii + 1) * z[ii + 1] - sumvec[ii + 1]) / lambda;
    if (right_sum >= 1 && left_sum < 1)
      eta = (1 / nn - 1 / ii) - (1 / ii) * sumvec[ii] / lambda;
      return (eta, ii);
    elseif (left_sum >= 1)
      # Need to decrease index, left sum is too large
      high_ind = ii - 1;
    else
      # Increase index, because right sum is too low
      low_ind = ii + 1;
    end    
  end
  ii = low_ind;
  eta = (1 / nn - 1 / ii) - (1 / ii) * sumvec[ii] / lambda;
  return (eta, ii);
end

# TestProjections()
#
# A unit test to verify that a number of simple special cases of
# projections work.
function TestProjections()
  test_tol = 1e-6;
  rho = 1.2;
  z = repmat(linspace(0, 1, 4), 4, 1)[:];
  x = MinimizeLinearSquareDivergence(z, rho);
  num_passed = 0;
  total_to_pass = 0;
  x_desired = [ 0.229583 
               0.0204175
               0.0      
               0.0      
               0.229583 
               0.0204175
               0.0      
               0.0      
               0.229583 
               0.0204175
               0.0      
               0.0      
               0.229583 
               0.0204175
               0.0      
               0.0  ];
  total_to_pass += 1;
  if (norm(x - x_desired) > test_tol)
    warn(string("Projection of 0, .33, .66, 1 vector failed."));
  else
    num_passed += 1;
  end
  z =  [-0.705131
        -1.6036  
        1.6838  
        -0.167923
        -0.514595
        0.488477 ];
  x = MinimizeLinearSquareDivergence(z, rho);
  x_desired = [  0.194745 
               0.722411 
               0.0      
               0.0      
               0.0828439
               0.0      
               ];
  total_to_pass += 1;
  if (norm(x - x_desired) > test_tol)
    warn(string("Random-ish string of z failed."));
  else
    num_passed += 1;
  end
  x = MinimizeLinearSquareDivergence(z, 200.0);
  total_to_pass += 1;
  if (norm(x - [0; 1; 0; 0; 0; 0]) > test_tol)
    warn(string("Large rho test failed."));
  else
    num_passed += 1;
  end
  x = MinimizeLinearSquareDivergence(z, 1e-14);
  total_to_pass += 1;
  if (norm(x - ones(6) / 6) > test_tol)
    warn(string("Very small rho test failed."));
  else
    num_passed += 1;
  end
  println("Passed ", num_passed, " of ", total_to_pass, " tests.");
end

end  # module SProj
