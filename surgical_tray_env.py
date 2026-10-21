import os
import numpy as np
from gymnasium import spaces
try:
    from gymnasium_robotics.core import MujocoRobotEnv
except ImportError:
    from gymnasium_robotics.envs.robot_env import MujocoRobotEnv


class SurgicalTrayEnv(MujocoRobotEnv):
    """
    Custom Surgical Tray Environment for Multi-Object, Multi-Tray workflows.
    """
    # Per-tool workflow stages, in order (Path 1 skips IN_CLEANING)
    STAGES = ["IN_STORAGE", "AT_SURGEON", "IN_USE", "AT_SURGEON_USED", "IN_CLEANING", "DONE"]
    HUMAN_BODIES = ("surgeon_body", "surgeon_arm", "assistant_body")

    # Table tops are at z=0.8 (trays are visual only); a tool below this has fallen off
    DROP_HEIGHT = 0.7
    # Max height above a tray site for a tool to count as placed on that tray
    ON_TRAY_HEIGHT = 0.05
    # Reward per completed stage; larger than any drop in distance shaping at a stage
    # hand-over (~1.2 m between trays), so completing a stage never lowers the reward
    PROGRESS_WEIGHT = 2.0

    def __init__(self, model_path="surgical_env.xml", workflow_path=2, reward_type="dense", **kwargs):
        self.objects = ["scalpel", "grasper", "scissors"]
        self.trays = {
            "IN_STORAGE": "site_tray_storage",
            "AT_SURGEON": "site_tray_surgeon",
            "IN_CLEANING": "site_tray_cleaning"
        }
        self.object_states = {obj: "IN_STORAGE" for obj in self.objects}
        
        # We track how long the surgeon has held it to simulate usage
        self.surgeon_timers = {obj: 0 for obj in self.objects}
        self.workflow_path = workflow_path
        self.reward_type = reward_type
        
        # Threshold for detecting if object is on a tray
        self.tray_threshold = 0.1
        
        initial_qpos = {
            "robot:panda0_joint1": 0.0,
            "robot:panda0_joint2": -0.785,
            "robot:panda0_joint3": 0.0,
            "robot:panda0_joint4": -2.356,
            "robot:panda0_joint5": 0.0,
            "robot:panda0_joint6": 1.57,
            "robot:panda0_joint7": 0.785,
            "robot:panda0_finger_joint1": 0.02,
            "robot:panda0_finger_joint2": 0.02,
        }

        # Sequence tracking
        self.target_sequence = list(self.objects)
        self.current_request_idx = 0

        # Ensure the model path is absolute if it's relative
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
            
        super().__init__(
            model_path=model_path,
            initial_qpos=initial_qpos,
            n_actions=9, # 7 joints + 2 fingers
            n_substeps=20,
            **kwargs
        )

        # Body ids for collision checks (human obstacle geoms are unnamed, so match by body)
        self._human_body_ids = {self.model.body(name).id for name in self.HUMAN_BODIES}
        self._robot_body_ids = {i for i in range(self.model.nbody) if self.model.body(i).name.startswith("panda0_")}
        self._tool_body_ids = {self.model.body(obj).id: obj for obj in self.objects}

        # Override observation space to include our custom obs
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype="float64"
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
            }
        )

    def _env_setup(self, initial_qpos):
        """Initializes the environment."""
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        import mujoco
        mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        """Randomizes the initial positions of the tools in Tray 1 (Storage)."""
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
            
        # Randomize Table Positions
        base_positions = {
            "table_storage": [-0.5, 0.2, 0.4],
            "table_surgeon": [0.5, 0.2, 0.4],
            "table_cleaning": [0.0, 0.45, 0.4]
        }
        for table_name, base_pos in base_positions.items():
            body_id = self.model.body(table_name).id
            offset_x = self.np_random.uniform(-0.05, 0.05)
            offset_y = self.np_random.uniform(-0.05, 0.05)
            self.model.body_pos[body_id] = [base_pos[0] + offset_x, base_pos[1] + offset_y, base_pos[2]]
            
        import mujoco
        mujoco.mj_forward(self.model, self.data)
            
        # Randomize object positions on Storage Tray (Tray 1)
        storage_site_id = self.model.site("site_tray_storage").id
        storage_pos = self.data.site_xpos[storage_site_id]
        
        for i, obj in enumerate(self.objects):
            # Tools are long along Y (~0.16 m) and thin along X (<= 0.04 m), so lay them
            # side by side along X; jitter stays inside the tray walls (inner 0.22 x 0.28 m)
            offset_x = (i - 1) * 0.06 + self.np_random.uniform(-0.01, 0.01)
            offset_y = self.np_random.uniform(-0.03, 0.03)

            obj_pos = np.array([
                storage_pos[0] + offset_x,
                storage_pos[1] + offset_y,
                storage_pos[2] + 0.01 # just above the tray surface
            ])
            
            joint_name = f"{obj}:joint"
            joint_qpos_addr = self.model.joint(joint_name).qposadr[0]
            
            # Set position
            self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] = obj_pos
            # Reset orientation (quaternion) to identity
            self.data.qpos[joint_qpos_addr+3:joint_qpos_addr+7] = [1, 0, 0, 0]
            
            # Reset object state
            self.object_states[obj] = "IN_STORAGE"
            self.surgeon_timers[obj] = 0

        # Randomize requested sequence
        self.np_random.shuffle(self.target_sequence)
        self.current_request_idx = 0
        self.start_time = self.data.time

        import mujoco
        mujoco.mj_forward(self.model, self.data)
        return True

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        action = action.copy()
        assert action.shape == (9,)
        
        # Panda 7 joints + 2 fingers: position actuators, so map [-1, 1] onto each
        # actuator's ctrlrange (raw actions would be clamped, e.g. joint4 to [-1, 0]
        # and the fingers to [0, 0.04])
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = low + (action + 1.0) * 0.5 * (high - low)

    def step(self, action):
        """Override step to update our state machine and dynamic obstacles."""
        _, _, terminated, truncated, info = super().step(action)
        import mujoco

        # 1. Dynamic Human Obstacles (kinematically driven: set qpos, clear qvel)
        # Pacing Assistant
        assistant_jnt_id = self.model.jnt("assistant_slider").id
        assistant_qpos_idx = self.model.jnt_qposadr[assistant_jnt_id]
        self.data.qpos[assistant_qpos_idx] = 0.2 * np.sin(self.data.time * 2.0)
        self.data.qvel[self.model.jnt_dofadr[assistant_jnt_id]] = 0.0

        # Sweeping Surgeon Arm
        arm_jnt_id = self.model.jnt("surgeon_arm_joint").id
        arm_qpos_idx = self.model.jnt_qposadr[arm_jnt_id]
        # Check if any tool is at surgeon to trigger reach
        is_tool_at_surgeon = any(state == "AT_SURGEON" for state in self.object_states.values())
        target_angle = -1.2 if is_tool_at_surgeon else 0.0
        current_angle = self.data.qpos[arm_qpos_idx]
        self.data.qpos[arm_qpos_idx] = current_angle + 0.1 * (target_angle - current_angle)
        self.data.qvel[self.model.jnt_dofadr[arm_jnt_id]] = 0.0

        # Update State Machine Logic
        self._update_object_states()

        # Propagate the qpos edits above (humans, teleported tools) to site positions and
        # contacts, then observe the post-update state so reward/success are not one step stale
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()

        sparse_reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        if isinstance(sparse_reward, np.ndarray):
            sparse_reward = float(sparse_reward[0])
        else:
            sparse_reward = float(sparse_reward)

        dense_reward = self._compute_dense_reward()

        info["sparse_reward"] = sparse_reward
        info["dense_reward"] = dense_reward
        info["is_success"] = self.is_success(obs["achieved_goal"], obs["desired_goal"])

        # Collision Check for binary failure: a human touching the robot or a tool it handles
        collision_detected = self._detect_human_collision()

        # Dropped tool check
        tool_dropped = False
        for obj in self.objects:
            obj_pos = self.data.site_xpos[self.model.site(f"{obj}_site").id]
            if obj_pos[2] < self.DROP_HEIGHT and self.object_states[obj] != "IN_USE":
                tool_dropped = True

        if self.reward_type == "binary":
            if collision_detected or tool_dropped:
                reward = -1.0
                terminated = True
            elif info["is_success"]:
                reward = 1.0
                terminated = True
            else:
                reward = 0.0
        else:
            reward = sparse_reward + dense_reward
            if collision_detected:
                reward -= 1.0 # Penalize collisions heavily

        if info["is_success"]:
            info["completion_time_seconds"] = self.data.time - getattr(self, "start_time", 0.0)

        info["collision"] = collision_detected
        info["dropped"] = tool_dropped

        return obs, reward, terminated, truncated, info

    def _detect_human_collision(self):
        for contact in self.data.contact[:self.data.ncon]:
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            for human, other in ((b1, b2), (b2, b1)):
                if human not in self._human_body_ids:
                    continue
                if other in self._robot_body_ids:
                    return True
                tool = self._tool_body_ids.get(other)
                if tool is not None and self.object_states[tool] != "IN_USE":
                    return True
        return False

    def _on_tray(self, obj_pos, tray_pos):
        """True if the tool rests on the tray (within its footprint, not carried above it)."""
        return (np.linalg.norm(obj_pos[:2] - tray_pos[:2]) < self.tray_threshold
                and obj_pos[2] < tray_pos[2] + self.ON_TRAY_HEIGHT)

    def _teleport(self, obj, pos):
        """Moves a free-floating tool and clears its velocity (otherwise gravity keeps accumulating)."""
        body_id = self.model.body(obj).id
        jnt_id = self.model.body_jntadr[body_id]
        qpos_idx = self.model.jnt_qposadr[jnt_id]
        dof_idx = self.model.jnt_dofadr[jnt_id]
        self.data.qpos[qpos_idx:qpos_idx+3] = pos
        self.data.qvel[dof_idx:dof_idx+6] = 0.0

    def _update_object_states(self):
        storage_pos = self.data.site_xpos[self.model.site("site_tray_storage").id]
        surgeon_pos = self.data.site_xpos[self.model.site("site_tray_surgeon").id]
        cleaning_pos = self.data.site_xpos[self.model.site("site_tray_cleaning").id]
        grip_pos = self.data.site_xpos[self.model.site("end_effector").id]

        for obj in self.objects:
            obj_pos = self.data.site_xpos[self.model.site(f"{obj}_site").id]
            current_state = self.object_states[obj]

            dist_to_grip = np.linalg.norm(obj_pos - grip_pos)

            if current_state == "IN_STORAGE":
                if self._on_tray(obj_pos, surgeon_pos):
                    self.object_states[obj] = "AT_SURGEON"

            elif current_state == "AT_SURGEON":
                # If robot releases it at surgeon tray, surgeon picks it up
                if dist_to_grip > 0.1 and self._on_tray(obj_pos, surgeon_pos):
                    self.object_states[obj] = "IN_USE"
                    self.surgeon_timers[obj] = 50 # Surgeon holds it for 50 steps

            elif current_state == "IN_USE":
                if self.surgeon_timers[obj] > 0:
                    self.surgeon_timers[obj] -= 1
                    # Teleport to surgeon hands
                    human_pos = self.data.site_xpos[self.model.site("surgeon_site").id]
                    self._teleport(obj, human_pos)
                else:
                    self.object_states[obj] = "AT_SURGEON_USED"
                    # Teleport back to surgeon tray
                    self._teleport(obj, surgeon_pos + np.array([0, 0, 0.05]))

            elif current_state == "AT_SURGEON_USED":
                if self.workflow_path == 1:
                    # Direct to storage
                    if self._on_tray(obj_pos, storage_pos):
                        self.object_states[obj] = "DONE"
                else:
                    # Path 2 requires cleaning first
                    if self._on_tray(obj_pos, cleaning_pos):
                        self.object_states[obj] = "IN_CLEANING"

            elif current_state == "IN_CLEANING":
                if self._on_tray(obj_pos, storage_pos):
                    self.object_states[obj] = "DONE"

        # If the currently requested tool has finished its full cycle, move to next tool
        if self.current_request_idx < len(self.target_sequence):
            current_req_tool = self.target_sequence[self.current_request_idx]
            if self.object_states[current_req_tool] == "DONE":
                self.current_request_idx += 1

    def _compute_dense_reward(self):
        """Calculates distance-based continuous rewards and safety penalties."""
        stages_per_tool = len(self.STAGES) - 1
        if self.current_request_idx >= len(self.target_sequence):
            return self.PROGRESS_WEIGHT * stages_per_tool * len(self.target_sequence) # Sequence complete

        req_tool = self.target_sequence[self.current_request_idx]
        req_pos = self.data.site_xpos[self.model.site(f"{req_tool}_site").id]
        grip_pos = self.data.site_xpos[self.model.site("end_effector").id]
        req_state = self.object_states[req_tool]

        # 1. Progress Reward: completed requests plus the requested tool's workflow stage
        progress = self.current_request_idx * stages_per_tool + self.STAGES.index(req_state)
        dense_reward = self.PROGRESS_WEIGHT * progress

        dist_to_tool = np.linalg.norm(grip_pos - req_pos)

        # 2. Approach, Grasp & Delivery Reward (only while the robot has to move the tool)
        target_site = None
        if req_state == "IN_STORAGE":
            target_site = "site_tray_surgeon"
        elif req_state == "AT_SURGEON_USED":
            target_site = "site_tray_storage" if self.workflow_path == 1 else "site_tray_cleaning"
        elif req_state == "IN_CLEANING":
            target_site = "site_tray_storage"

        if target_site is not None:
            dense_reward -= dist_to_tool
            target_pos = self.data.site_xpos[self.model.site(target_site).id]
            dist_to_target = np.linalg.norm(req_pos[:2] - target_pos[:2]) # 2D distance to tray

            if dist_to_tool < 0.05: # Gripper is close/grasping the tool
                dense_reward += 0.5 # Grasp bonus
                dense_reward -= dist_to_target # Incentivize moving to target tray
        elif req_state == "AT_SURGEON":
            # Hand-over needs the gripper > 0.1 m from the tool; reward letting go
            # instead of pulling the gripper onto it (which blocks the hand-over)
            dense_reward -= max(0.0, 0.1 - dist_to_tool)
        # IN_USE: the surgeon holds the tool, so don't pull the gripper towards the surgeon

        # 3. Safety Penalties
        for obj in self.objects:
            obj_pos = self.data.site_xpos[self.model.site(f"{obj}_site").id]

            # Drop Penalty (fell below the table tops)
            if obj_pos[2] < self.DROP_HEIGHT and self.object_states[obj] != "IN_USE":
                dense_reward -= 1.0

            # Wrong Tool Manipulation Penalty
            if obj != req_tool and self.object_states[obj] != "IN_USE":
                dist_to_wrong_tool = np.linalg.norm(grip_pos - obj_pos)
                if dist_to_wrong_tool < 0.05:
                    dense_reward -= 0.5

        return dense_reward

    def _get_site_velocity(self, site_name):
        import mujoco
        site_id = self.model.site(site_name).id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0)
        return vel[3:], vel[:3]  # mj_objectVelocity returns [rotational, translational]

    def _get_obs(self):
        # Panda observations
        grip_pos = self.data.site_xpos[self.model.site("end_effector").id]
        grip_velp, grip_velr = self._get_site_velocity("end_effector")
        
        # Finger states
        # Index qpos/qvel by the joint's address, not its id (the tools' free joints come first)
        finger_ids = [self.model.jnt(f"robot:panda0_finger_joint{i}").id for i in (1, 2)]
        gripper_state = self.data.qpos[self.model.jnt_qposadr[finger_ids]].copy()
        gripper_vel = self.data.qvel[self.model.jnt_dofadr[finger_ids]].copy()
        
        object_features = []
        achieved_goal = []
        for obj in self.objects:
            obj_pos = self.data.site_xpos[self.model.site(f"{obj}_site").id]
            obj_rot = self.data.site_xmat[self.model.site(f"{obj}_site").id].flatten()
            obj_velp, obj_velr = self._get_site_velocity(f"{obj}_site")
            state = self.object_states[obj]
            # State is expanded to 6 dimension one-hot
            # IN_STORAGE, AT_SURGEON, IN_USE, AT_SURGEON_USED, IN_CLEANING, DONE
            one_hot_state = np.zeros(6)
            if state == "IN_STORAGE": one_hot_state[0] = 1
            elif state == "AT_SURGEON": one_hot_state[1] = 1
            elif state == "IN_USE": one_hot_state[2] = 1
            elif state == "AT_SURGEON_USED": one_hot_state[3] = 1
            elif state == "IN_CLEANING": one_hot_state[4] = 1
            elif state == "DONE": one_hot_state[5] = 1
            
            object_features.extend([obj_pos, obj_rot, obj_velp, obj_velr, one_hot_state])
            achieved_goal.extend(one_hot_state) # Track states for goal success
            
        # One-hot encoding of currently requested tool
        request_one_hot = np.zeros(len(self.objects))
        if self.current_request_idx < len(self.target_sequence):
            req_tool = self.target_sequence[self.current_request_idx]
            req_idx = self.objects.index(req_tool)
            request_one_hot[req_idx] = 1.0
            
        obs = np.concatenate([
            grip_pos, grip_velp, gripper_state, gripper_vel,
            np.concatenate(object_features),
            request_one_hot
        ])
        
        # Desired goal: All objects in DONE state
        desired_goal = self._sample_goal()

        return {
            "observation": obs.copy(),
            "achieved_goal": np.array(achieved_goal).copy(),
            "desired_goal": desired_goal.copy(),
        }

    def is_success(self, achieved_goal, desired_goal):
        """Returns True only if all objects have completed the full cycle (are DONE)."""
        # achieved_goal contains the one-hot states of all objects
        return np.allclose(achieved_goal, desired_goal)

    def _is_success(self, achieved_goal, desired_goal):
        """Alias for is_success for newer gymnasium_robotics versions."""
        return self.is_success(achieved_goal, desired_goal)

    def _sample_goal(self):
        """Returns the desired state where all objects are DONE."""
        # 3 objects, each has a 6-dim one-hot state
        # "DONE" is index 5
        goal = np.zeros(18)
        goal[5] = 1.0   # object 1 DONE
        goal[11] = 1.0  # object 2 DONE
        goal[17] = 1.0  # object 3 DONE
        return goal

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse reward: 1.0 if full cycle complete, 0.0 otherwise."""
        # For batched environments, iterate over them
        if len(achieved_goal.shape) == 2:
            return np.array([1.0 if self.is_success(ag, dg) else 0.0 for ag, dg in zip(achieved_goal, desired_goal)])
        else:
            return 1.0 if self.is_success(achieved_goal, desired_goal) else 0.0
