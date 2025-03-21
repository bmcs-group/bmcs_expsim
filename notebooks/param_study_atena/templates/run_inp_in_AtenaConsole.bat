cd AtenaCalculationModified

cmd /K start /B "ATENA calculation" %AtenaConsole64% /M CCStructuresCreep /execute /catch_fp_instructs /o "atena_study_basename.inp" "atena_study_basename.out" "atena_study_basename.msg" "atena_study_basename.err" /num_unused_threads=2  /num_iters_per_thread=0

