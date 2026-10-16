import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

def main(args=None):
    rclpy.init(args=args)
    # 建立一個名為 ros2_talker_node 的節點
    node = Node('ros2_talker_node')
    # 建立一個發布者，主題名稱叫 'chatter'
    pub = node.create_publisher(String, 'chatter', 10)
    
    msg = String()
    i = 0
    
    try:
        while rclpy.ok():
            msg.data = f'Hello World: {i}'
            pub.publish(msg)
            node.get_logger().info(f'Publishing: "{msg.data}"')
            i += 1
            time.sleep(0.5) # 每 0.5 秒發布一次
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()