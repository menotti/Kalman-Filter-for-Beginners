# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct C:\Users\menotti\Workspace\kalman\project_1\vitis\design_1_wrapper\platform.tcl
# 
# OR launch xsct and run below command.
# source C:\Users\menotti\Workspace\kalman\project_1\vitis\design_1_wrapper\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {design_1_wrapper}\
-hw {C:\Users\menotti\Workspace\kalman\project_1\design_1_wrapper.xsa}\
-out {C:/Users/menotti/Workspace/kalman/project_1/vitis}

platform write
domain create -name {standalone_ps7_cortexa9_0} -display-name {standalone_ps7_cortexa9_0} -os {standalone} -proc {ps7_cortexa9_0} -runtime {cpp} -arch {32-bit} -support-app {empty_application}
platform generate -domains 
platform active {design_1_wrapper}
domain active {zynq_fsbl}
domain active {standalone_ps7_cortexa9_0}
platform generate -quick
domain active {zynq_fsbl}
platform generate
bsp reload
bsp reload
domain active {standalone_ps7_cortexa9_0}
bsp config dependency_flags "-MMD -MP"
bsp config dependency_flags "-MMD -MP"
bsp config extra_compiler_flags "-mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard -nostartfiles -g -Wall -Wextra -fno-tree-loop-distribute-patterns -lm"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_ps7_cortexa9_0 
