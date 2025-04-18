from app.utils.robot.openpi_client import image_tools
from app.utils.robot.openpi_client import websocket_client_policy
import numpy as np
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)


num_steps = 2
for step in range(num_steps):
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    state = np.random.rand(6)
    task_instruction = "Move the robot to the target position"

    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 480, 640)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 480, 640)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }


    action_chunk = client.infer(observation)["actions"]

    print(action_chunk)