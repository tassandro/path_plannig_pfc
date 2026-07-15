import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc/ros2_ws/install/ugv_bridge'
