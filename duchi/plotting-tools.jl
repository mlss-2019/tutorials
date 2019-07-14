# plotting-tools.jl
#
# Generic methods for plotting times to solution

using PyPlot;
using Distributions;

include("utilities.jl");

# max_time =
#  PlotTimesToEpsilon(sg_gaps, trunc_gaps, prox_gaps;
#                     stepsizes, eps_threshold, confidence_width)
#
# Plots the median time to epsilon accuracy (given by eps_threshold)
# for the results in the 3-way tensors sg_gaps, trunc_gaps, prox_gaps.
#
# Each of the *_gaps tensors is of size n1-by-n2-by-n3, where n1 is
# the number of objective computations, n2 the number of different
# stepsizes, and n3 the number of different tests (algorithm runs) per
# stepsize.
#
# If confidence_width > 0, then plots the [.5 - confidence_width, .5 +
# confidence_width] quantiles of the different methods' performances
# as well. Returns the maximum time of all methods.
function PlotTimesToEpsilon(sg_gaps::Array{Float64,3},
                            trunc_gaps::Array{Float64,3},
                            prox_gaps::Array{Float64,3};
                            stepsizes::Array{Float64,1} = Array{Float64,1}(),
                            eps_threshold::Float64 = 1e-2,
                            confidence_width::Float64 = 0.0,
                            plot_prox::Bool = true)
  (num_objs_computed, num_stepsizes, num_tests) =
    size(sg_gaps);
  median_time_sg = zeros(num_stepsizes);
  median_time_prox = zeros(num_stepsizes);
  median_time_trunc = zeros(num_stepsizes);
  upper_time_sg = zeros(num_stepsizes);
  upper_time_prox = zeros(num_stepsizes);
  upper_time_trunc = zeros(num_stepsizes);
  lower_time_sg = zeros(num_stepsizes);
  lower_time_prox = zeros(num_stepsizes);
  lower_time_trunc = zeros(num_stepsizes);
  confidence_width = max(min(confidence_width, .5), 0.0);
  if (isempty(prox_gaps))
    plot_prox = false;
  end
  if (isempty(stepsizes))
    stepsizes = logspace(0,1, num_stepsizes);
  end
  for step_ind = 1:num_stepsizes
    times_to_eps = zeros(num_tests);
    for ii = 1:num_tests
      ind = findfirst(sg_gaps[:, step_ind, ii] .<= eps_threshold);
      times_to_eps[ii] = (ind != nothing ? ind : num_objs_computed);
    end
    median_time_sg[step_ind] = median(times_to_eps);
    upper_time_sg[step_ind] = quantile(times_to_eps, .5 + confidence_width);
    lower_time_sg[step_ind] = quantile(times_to_eps, .5 - confidence_width);
    for ii = 1:num_tests
      ind = findfirst(trunc_gaps[:, step_ind, ii] .<= eps_threshold);
      times_to_eps[ii] = (ind != nothing ? ind : num_objs_computed);
    end
    median_time_trunc[step_ind] = median(times_to_eps);
    upper_time_trunc[step_ind] = quantile(times_to_eps, .5 + confidence_width);
    lower_time_trunc[step_ind] = quantile(times_to_eps, .5 - confidence_width);
    if (plot_prox)
      for ii = 1:num_tests
        ind = findfirst(prox_gaps[:, step_ind, ii] .<= eps_threshold);
        times_to_eps[ii] = (ind != nothing ? ind : num_objs_computed);
      end
      median_time_prox[step_ind] = median(times_to_eps);
      upper_time_prox[step_ind] = quantile(times_to_eps, .5 + confidence_width);
      lower_time_prox[step_ind] = quantile(times_to_eps, .5 - confidence_width);
    end
  end

  figure();
  semilogx(stepsizes, median_time_sg, label="SGM", "rv:");
  semilogx(stepsizes, median_time_trunc, label="Truncated",
           "ko-.");
  if (plot_prox)
    semilogx(stepsizes, median_time_prox, label="Proximal",
             "bs-");
  end
  if (confidence_width > 0)
    fill_between(stepsizes, upper_time_sg, lower_time_sg,
                 color = (1,0,0), alpha = .3);
    fill_between(stepsizes, upper_time_trunc, lower_time_trunc,
                 color = (0,0,0), alpha = .25);
    if (plot_prox)
      fill_between(stepsizes, upper_time_prox, lower_time_prox,
                   color = (0,0,1), alpha = .25);
    end
  end
  return max(maximum(upper_time_sg),
             maximum(upper_time_trunc),
             (plot_prox ? maximum(upper_time_prox) : 0));
end

# # PlotSingleExperimentResult(filename::String,
# #                            optimality_gap::Float64,
# #                            confidence_width::Float64;
# #                            ytick_scale = 100)
# #
# # Plots the time to epsilon = optimality_gap accuracy for the
# # experiment stored at the given filename. Rescales the y scale by the
# # given scaling, and plots a confidence width.
# function PlotSingleExperimentResult(filename::String,
#                                     optimality_gap::Float64,
#                                     confidence_width::Float64;
#                                     ytick_scale = 100)
#   (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
#     ReadGaps(filename=filename);
#   max_time = PlotTimesToEpsilon(sg_gaps, trunc_gaps, prox_gaps,
#                                 eps_threshold = optimality_gap,
#                                 confidence_width = confidence_width,
#                                 stepsizes = stepsizes);
#   legend(fontsize=16);
#   axis([stepsizes[1] / 2, 2 * stepsizes[end],
#         0, 1.1 * max_time]);
#   ytick_locs = copy(yticks()[1]);
#   yticks(ytick_locs, round.(Int64, ytick_locs * ytick_scale));
# end

# # PlotSingleMatrixCompletionResult(filename::String,
# #                                           optimality_gap::Float64,
# #                                           confidence_width::Float64;
# #                                           ytick_scale = 100,
# #                                           parameter_recovery = true,
# #                                           plot_prox = true)
# #
# # Plots the results of a single matrix completion experiment.
# function PlotSingleMatrixCompletionResult(filename::String,
#                                           optimality_gap::Float64,
#                                           confidence_width::Float64;
#                                           ytick_scale = 100,
#                                           parameter_recovery = true,
#                                           plot_prox = true)
#   (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
#     ReadGapsFourWay(filename=filename);
#   ind_to_plot = (parameter_recovery ? 2 : 1);
#   max_time = PlotTimesToEpsilon(sg_gaps[:, ind_to_plot, :, :],
#                                 trunc_gaps[:, ind_to_plot, :, :],
#                                 prox_gaps[:, ind_to_plot, :, :],
#                                 eps_threshold = optimality_gap,
#                                 confidence_width = confidence_width,
#                                 stepsizes = stepsizes,
#                                 plot_prox = plot_prox);
#   legend(fontsize=16);
#   axis([stepsizes[1] / 2, 2 * stepsizes[end],
#         0, 1.1 * max_time]);
#   ytick_locs = copy(yticks()[1]);
#   # yticks(ytick_locs, round.(Int64, ytick_locs * ytick_scale));
# end

# function PlotMultipleRecoveries(filenames::Array{String},
#                                 optimality_gap::Float64;
#                                 parameter_recovery = true,
#                                 rescale = false,
#                                 m::Int64 = 1, n::Int64 = 1)
#   # Construct mean_recovery_times matrix to be of size
#   # length(filenames)-by-number of stepsizes.
#   (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
#     ReadGaps(filename=filenames[1]);
#   # The 3-way tensor for each of stochastic gradient, truncated, and proximal
#   # point.
#   mean_recovery_times = zeros(length(filenames), length(stepsizes), 3);
#   max_num_objs = 0;
#   max_num_tests = 0;
#   for ii = 1:length(filenames)
#     (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
#       ReadGaps(filename=filenames[ii]);
#     all_gaps = Array{Array{Float64}}(3);
#     all_gaps[1] = sg_gaps;
#     all_gaps[2] = trunc_gaps;
#     all_gaps[3] = prox_gaps;
#     for jj = 1:3
#       all_gaps[jj][isnan.(all_gaps[jj])] = Inf;
#     end
#     if (length(size(sg_gaps)) == 4)
#       # The gaps are from an experiment recording both parameter error and
#       # objective values. The size of each *_gaps tensor is
#       # (num_objs_computed, 2, num_initial_stepsizes, num_tests).
#       (num_objs_computed, temp, num_initial_stepsizes, num_tests) =
#         size(sg_gaps);
#       max_num_objs = max(num_objs_computed, max_num_objs);
#       max_num_tests = max(num_tests, max_num_tests);
#       gap_ind = 1;
#       if (parameter_recovery)
#         gap_ind = 2;
#       end
#       for jj = 1:num_initial_stepsizes
#         for kk = 1:num_tests
#           for method_ind = 1:3
#             if (rescale)
#               final_err = all_gaps[method_ind][end, gap_ind, jj, kk];
#               if (parameter_recovery)
#                 final_err /= (m * n);
#               end
#               mean_recovery_times[ii, jj, method_ind] +=
#                 2 * atan(final_err) / pi;
#             else
#               first_ind =
#                 findfirst(all_gaps[method_ind][:, gap_ind, jj, kk]
#                           .< optimality_gap);
#               first_ind = (first_ind == 0 ? num_objs_computed : first_ind);
#               mean_recovery_times[ii, jj, method_ind] += first_ind;
#             end
#           end
#         end
#       end
#     else
#       # The gaps are from an experiment recording only objective values (or
#       # gaps). The size of each *_gaps tensor is (num_objs_computed, 2,
#       # num_inits, num_tests).
#       error("Have not implemented objective value plots yet.");
#     end
#   end
#   mean_recovery_times = mean_recovery_times / (max_num_objs * max_num_tests);

#   # Now show figures for each of SGM, Truncated, and Prox. First, though, find
#   # the actual range of values to plot for each of them.
#   max_recoveries = maximum(mean_recovery_times);
#   largest_file_ind = 0;
#   for ii = 1:length(filenames)
#     if (any(mean_recovery_times[ii, :, :] .< max_recoveries))
#       largest_file_ind = ii;
#     end
#   end
#   smallest_stepsize_ind = length(stepsizes);
#   largest_stepsize_ind = 1;
#   for ii = 1:length(filenames)
#     for method_ind = 1:3
#       first_ind = findfirst(mean_recovery_times[ii, :, method_ind]
#                             .< max_recoveries);
#       smallest_stepsize_ind = (first_ind == 0 ? smallest_stepsize_ind :
#                                min(first_ind, smallest_stepsize_ind));
#       last_ind = findlast(mean_recovery_times[ii, :, method_ind]
#                           .< max_recoveries);
#       largest_stepsize_ind = (last_ind == 0 ? largest_stepsize_ind :
#                               max(largest_stepsize_ind, last_ind));
#     end
#   end
#   if (smallest_stepsize_ind > largest_stepsize_ind)
#     smallest_stepsize_ind = 1;
#     largest_stepsize_ind = length(stepsizes);
#   end
#   if (smallest_stepsize_ind > 1)
#     smallest_stepsize_ind -= 1;
#   end
#   if (largest_stepsize_ind < length(stepsizes))
#     largest_stepsize_ind += 1;
#   end
#   println("Showing stepsizes between ",
#           stepsizes[smallest_stepsize_ind], " and ",
#           stepsizes[largest_stepsize_ind],
#           " (indices ", smallest_stepsize_ind,
#           " to ", largest_stepsize_ind, ")");
#   titles = ["SGM", "Truncated", "Prox"];
#   for ii = 1:3
#     figure();
#     stepsize_inds = smallest_stepsize_ind:largest_stepsize_ind;
#     imshow(1 - mean_recovery_times[:, stepsize_inds, ii],
#            cmap="gray", interpolation="nearest");
#     title(titles[ii]);
#     rounded_stepsizes =
#       RoundToTwoSignificantDigits(stepsizes[stepsize_inds]);
#     xticks(0:2:(length(stepsize_inds) - 1),
#            rounded_stepsizes[1:2:end]);
#     # xticks(rounded_stepsizes);
#     # xticks(0:2:(length(stepsizes)-1), rounded_stepsizes);
#     xlabel("Stepsizes");
#   end
# end

# function RoundToTwoSignificantDigits(v::Vector{Float64})
#   return map((x) -> (isinteger(x) ? Int(x) : x),
#              map((x) -> round(x, -(floor(Int64, log10(x)) - 1)),
#                  v));
# end

# # PlotMCHingeResults(; optimality_gaps = [.01, .03, .2],
# #                      ytick_scale = 100)
# #
# # Plots the hinge results and saves them in pdf files. There are three
# # experiments: one with no noise (positive margin), one with a little
# # noise (smallnoise), flipping .01 fraction of the examples, and one
# # with large noise (bignoise), flipping .1 fraction of the examples.
# function PlotMCHingeResults(;
#                             optimality_gaps = [.01, .03, .2],
#                             ytick_scale = 100)
#   close("all");
#   filenames = [ "hinge-m1000-n10-k10-margin.txt",
#                 "hinge-m1000-n10-k10-smallnoise.txt",
#                 "hinge-m1000-n10-k10-bignoise.txt" ];
#   savefilenames = [ "hinge-m1000-n10-k10-margin.pdf",
#                     "hinge-m1000-n10-k10-smallnoise.pdf",
#                     "hinge-m1000-n10-k10-bignoise.pdf" ];
#   confidence_widths = .25 * ones(length(filenames));
#   for ii = 1:length(filenames)
#     PlotSingleExperimentResult(filenames[ii], optimality_gaps[ii],
#                                confidence_widths[ii],
#                                ytick_scale = ytick_scale);
#     savefig(savefilenames[ii], bbox_inches="tight");
#   end
# end

# # PlotLinregResults(; ytick_scale = 100)
# #
# # Plots the linear regression results and saves them in pdf files.
# function PlotLinregResults(; ytick_scale = 100)
#   close("all");
#   filenames = [ "absreg-m100-n10-nonoise.txt",
#                 "absreg-m100-n10-noise.txt",
#                 "linreg-m100-n10-nonoise.txt",
#                 "linreg-m100-n10-smallnoise.txt",
#                 "linreg-m100-n10-bignoise.txt" ];
#   savefilenames = ["absreg-m100-n10-nonoise.pdf",
#                    "absreg-m100-n10-noise.pdf",
#                    "linreg-m100-n10-nonoise.pdf",
#                    "linreg-m100-n10-smallnoise.pdf",
#                    "linreg-m100-n10-bignoise.pdf" ];
#   optimality_gaps = [1e-2, 1e-2, 1e-4, 1e-3, .5e-1];
#   confidence_widths = [.25, .3, .4, .25, .25];
#   for ii = 1:length(filenames)
#     PlotSingleExperimentResult(filenames[ii], optimality_gaps[ii],
#                                confidence_widths[ii],
#                                ytick_scale = ytick_scale);
#     savefig(savefilenames[ii], bbox_inches="tight");
#   end
# end

# # PlotPhaseRetrievalResults(; ytick_scale = 50)
# #
# # Plots the phase retrieval results for a phase retrieval experiment with
# # Poisson noise and poisson likelihood objective.
# function PlotPhaseRetrievalResults(; ytick_scale = 50)
#   close("all");
#   filenames = [ "pr-poisson-m400-n20.txt",
#                 "pr-poisson-m800-n100.txt" ];
#   savefilenames = [ "pr-poisson-m400-n20.pdf",
#                     "pr-poisson-m800-n100.pdf" ];
#   ms = [ 400, 800 ];
#   ns = [ 20, 100 ];
#   optimality_gaps = 2 * sqrt.(ns) .* sqrt.(ns ./ ms);
#   confidence_widths = .25 * ones(length(filenames));
#   for ii = 1:length(filenames)
#     PlotSingleExperimentResult(filenames[ii], optimality_gaps[ii],
#                                confidence_widths[ii],
#                                ytick_scale = ytick_scale);
#     savefig(savefilenames[ii], bbox_inches="tight");
#   end
# end

# # (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
# #   ReadGaps(; savepath::String = base_savepath,
# #              filename::String = "stepsize.txt")
# #
# # Reads the function value gaps for the different methods, returning 3-way
# # tensors of the gaps.
# function ReadGaps(;
#                   savepath::String = base_savepath,
#                   filename::String = "stepsize.txt")
#   result = readdlm(string(base_savepath, "/", filename));
#   num_objs_computed = round(Int64, result[1]);
#   # if (round(Int64, result[2]) - result[2] == 0)
#   #   return ReadGapsFourWay(savepath=savepath, filename=filename);
#   # end
#   num_stepsizes = round(Int64, result[2]);
#   num_tests = round(Int64, result[3]);
#   println("Reading file with ", num_objs_computed,
#           " objective steps, ", num_stepsizes, " stepsizes, ",
#           num_tests, " tests per method.");
#   stepsize_inds = 4:(num_stepsizes + 3);
#   stepsizes = result[stepsize_inds];
#   inds_per_method = num_objs_computed * num_stepsizes * num_tests;
#   curr_ind = stepsize_inds[end] + 1;
#   sg_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                     (num_objs_computed, num_stepsizes,
#                      num_tests));
#   curr_ind = curr_ind + inds_per_method;
#   trunc_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                        (num_objs_computed, num_stepsizes,
#                         num_tests));
#   curr_ind = curr_ind + inds_per_method;
#   prox_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                     (num_objs_computed, num_stepsizes,
#                      num_tests));
#   return (sg_gaps, trunc_gaps, prox_gaps, stepsizes);
# end

# # (sg_gaps, trunc_gaps, prox_gaps, stepsizes) =
# #   ReadGapsFourWay(; savepath::String = base_savepath,
# #                     filename::String = "stepsize.txt")
# #
# # Reads a 4-way tensor of the objective value gaps, where the second column of
# # each set of gaps includes parameter recovery errors.
# function ReadGapsFourWay(;
#                          savepath::String = base_savepath,
#                          filename::String = "stepsize.txt")
#   result = readdlm(string(base_savepath, "/", filename));
#   num_objs_computed = round(Int64, result[1]);
#   assert(convert(Int64, result[2]) == 2);
#   num_stepsizes = round(Int64, result[3]);
#   num_tests = round(Int64, result[4]);
#   println("Reading file with ", num_objs_computed,
#           " objective steps, ", num_stepsizes, " stepsizes, ",
#           num_tests, " tests per method.");
#   stepsize_inds = 5:(num_stepsizes + 4);
#   stepsizes = result[stepsize_inds];
#   inds_per_method = 2 * num_objs_computed * num_stepsizes * num_tests;
#   curr_ind = stepsize_inds[end] + 1;
#   sg_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                     (num_objs_computed, 2, num_stepsizes,
#                      num_tests));
#   curr_ind = curr_ind + inds_per_method;
#   trunc_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                        (num_objs_computed, 2, num_stepsizes,
#                         num_tests));
#   curr_ind = curr_ind + inds_per_method;
#   prox_gaps = reshape(result[curr_ind:(curr_ind + inds_per_method - 1)],
#                     (num_objs_computed, 2, num_stepsizes,
#                      num_tests));
#   return (sg_gaps, trunc_gaps, prox_gaps, stepsizes);
# end

