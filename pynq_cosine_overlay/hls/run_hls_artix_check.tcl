open_project -reset cosine_skip_hls_artix_check
set_top cosine_skip
add_files src/cosine_skip.cpp
add_files src/cosine_skip.h
add_files -tb src/cosine_skip_tb.cpp
open_solution -reset solution1 -flow_target vivado
set_part {xc7a50tcsg324-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
