set script_dir [file normalize [file dirname [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set build_dir [file normalize [file join $root_dir build]]
set proj_dir [file normalize [file join $build_dir vivado_proj]]
set artifacts_dir [file normalize [file join $root_dir artifacts]]
set ip_repo [file normalize [file join $root_dir hls cosine_skip_hls solution1 impl ip]]

file mkdir $build_dir
file mkdir $artifacts_dir

if {![file exists $ip_repo]} {
    error "HLS IP repo not found: $ip_repo. Run Vitis HLS first."
}

create_project -force cosine_overlay $proj_dir -part xc7z020clg400-1
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog

create_bd_design "cosine_overlay"

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells processing_system7_0]

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
    [get_bd_cells processing_system7_0]

set cosine_vlnv [lindex [get_ipdefs -all -filter {NAME == cosine_skip}] 0]
if {$cosine_vlnv eq ""} {
    error "cosine_skip IP was not found in catalog."
}
create_bd_cell -type ip -vlnv $cosine_vlnv cosine_skip_0

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins cosine_skip_0/ap_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins cosine_skip_0/ap_rst_n]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Slave "/cosine_skip_0/s_axi_control" Clk_master "/processing_system7_0/FCLK_CLK0" Clk_slave "/processing_system7_0/FCLK_CLK0" Clk_xbar "/processing_system7_0/FCLK_CLK0"} \
    [get_bd_intf_pins cosine_skip_0/s_axi_control]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]
connect_bd_intf_net [get_bd_intf_pins cosine_skip_0/m_axi_gmem0] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins cosine_skip_0/m_axi_gmem1] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

foreach clk_pin [get_bd_pins -of_objects [get_bd_cells smartconnect_0] -filter {TYPE == clk}] {
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $clk_pin
}
foreach rst_pin [get_bd_pins -of_objects [get_bd_cells smartconnect_0] -filter {TYPE == rst}] {
    connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] $rst_pin
}

assign_bd_address
validate_bd_design
save_bd_design

set wrapper_file [make_wrapper -files [get_files $proj_dir/cosine_overlay.srcs/sources_1/bd/cosine_overlay/cosine_overlay.bd] -top]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set bit_file [file normalize [file join $proj_dir cosine_overlay.runs impl_1 cosine_overlay_wrapper.bit]]
set hwh_file [file normalize [file join $proj_dir cosine_overlay.gen sources_1 bd cosine_overlay hw_handoff cosine_overlay.hwh]]

if {![file exists $bit_file]} { error "Bitstream not found: $bit_file" }
if {![file exists $hwh_file]} { error "HWH not found: $hwh_file" }

file copy -force $bit_file [file join $artifacts_dir cosine_overlay.bit]
file copy -force $hwh_file [file join $artifacts_dir cosine_overlay.hwh]
puts "Generated: [file join $artifacts_dir cosine_overlay.bit]"
puts "Generated: [file join $artifacts_dir cosine_overlay.hwh]"
exit
