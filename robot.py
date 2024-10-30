import socket
from ipaddress import ip_address
import lebai_sdk
import lebai_sdk.lebai_sdk

class RoboticArm:
    def __init__(self, ip_address=None, port=None):
        # 机械臂sdk初始化
        lebai_sdk.init()
        
        # ip地址初始化
        if not ip_address:
            self.ip_address = lebai_sdk.discover_devices(1)[0]['ip']
        else:
            self.ip_address = ip_address
        self.port = port

        # 相关参数初始化
        self.acceleration = 0.5 # 关节加速度 (rad/s2)
        self.velocity = 0.2 #关节速度 (rad/s)
        self.time_running = 0 # 运动时间 (s)。 当 t > 0 时，参数速度 v 和加速度 a 无效
        self.radius = 0 #交融半径 (m)。用于指定路径的平滑效果

        # 对象初始化
        self.lebai = None

    def update_motion_parameters(self,acceleration,velocity,time_running,radius):
        self.acceleration = acceleration
        self.velocity = velocity
        self.time_running = time_running
        self.radius = radius

    def get_position(self):
        position = self.lebai.get_kin_data()
        return position['actual_tcp_pose']

    def connect(self):
        """建立与机械臂的网络连接"""
        try:
            self.lebai = lebai_sdk.connect(self.ip_address,False)
            self.lebai.start_sys()
            print(f"已连接到机械臂：{self.ip_address}")
        except Exception as e:
            print(f"连接机械臂失败：{e}")

    def move_command(self, joint_pose):
        # mv_pose = {"x"}
        """向机械臂发送命令"""
        try:
            self.lebai.movej(joint_pose,
                             self.acceleration,
                             self.velocity,
                             self.time_running,
                             self.radius)
            print(f"机械臂运动位置为：：{joint_pose}")
            self.lebai.wait_move()
        except Exception as e:
            print(f"发送命令失败：{e}")

    def disconnect(self):
        """断开与机械臂的连接"""
        self.lebai.stop_sys()
        print("已断开与机械臂的连接")

    # def move_to_cordinate(self, cpose):
    #     # cpose = [x, y, z, Rz, Ry, Rx]
    #     jpose = self.lebai.kinematics_forward(cpose)
    #     print(f"jpose：{jpose}")
    #     self.move_command(jpose)

if __name__ == "__main__":
    robotic_arm = RoboticArm()
    print(robotic_arm.ip_address)

    cartesian_pose = {'x' : -0.3, 
                      'y' : -0.1, 
                      'z' : 0.2, 
                      'rz' : -99/180*3.14, 
                      'ry' : 0, 
                      'rx' : 99/180*3.14}
    
    robotic_arm.connect()
    jpose = robotic_arm.lebai.kinematics_inverse(cartesian_pose)
    print(f"kinematics_inverse：{jpose}")
    robotic_arm.move_command(jpose)
    print("Now pose", robotic_arm.get_position())
    # cpose = [-0.4, -0.1, 0.2, -90/180*3.14, 0, 90/180*3.14]
    # print(f"set go pose：{cpose}")
    # # robotic_arm.move_to_cordinate(cpose)
    # cur_pose = robotic_arm.lebai.kinematics_inverse(robotic_arm.get_position()) # 
    # print(robotic_arm.lebai.kinematics_inverse(cpose, cur_pose))
    # # robotic_arm.move_command([0,-1.0,1.05,0,1.57,0])
    # print("Now pose", robotic_arm.get_position())