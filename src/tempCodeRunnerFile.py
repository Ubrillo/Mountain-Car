# === PID BASED FORCE CORRECTION NEAR GOAL ===
                if abs(position - goal_position_valley) < 0.2:
                    error = goal_position_valley - position
                    pid_output = pid.compute(error, dt=1.0)
                    applied_force = np.clip(pid_output, -1, 1) * force
                else:
                    applied_force = action * force