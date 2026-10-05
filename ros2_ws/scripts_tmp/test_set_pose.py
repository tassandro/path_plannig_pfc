import rclpy
from rclpy.node import Node
from robot_localization.srv import SetPose

rclpy.init()
node = Node('test_set_pose_client')
cli = node.create_client(SetPose, '/j100_0000/set_pose')
cli.wait_for_service(timeout_sec=5.0)
req = SetPose.Request()
req.pose.header.frame_id = 'odom'
req.pose.pose.pose.position.x = 16.0
req.pose.pose.pose.position.y = -2.0
req.pose.pose.pose.position.z = 0.1
req.pose.pose.pose.orientation.w = 1.0
req.pose.pose.covariance = [0.001 if i in (0, 7, 14, 21, 28, 35) else 0.0 for i in range(36)]
future = cli.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
print('done', future.result())
rclpy.shutdown()
