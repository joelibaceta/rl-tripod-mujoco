import math
import time
import mujoco
import mujoco.viewer

# Parámetros de la marcha
STEP_START  = 1.0              # segundos antes de empezar a caminar
STEP_PERIOD = 4.0              # duración de una zancada completa (s)
omega       = 2*math.pi / STEP_PERIOD
A_HIP       = 30.0             # amplitud de caderas (grados)
A_KNEE      = 20.0             # amplitud de rodillas (grados)
PHASES      = [0, 2*math.pi/3, 4*math.pi/3]
SLOWDOWN    = 0.1                # para controlar fps reales

# 1) Carga modelo y datos
model = mujoco.MjModel.from_xml_path("tripod_model.xml")
data  = mujoco.MjData(model)



# 3) Abre el viewer pasivo
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 4) IDs de los servos de posición
    act_names = [
      "hip1_servo","knee1_servo",
      "hip2_servo","knee2_servo",
      "hip3_servo","knee3_servo"
    ]
    act_ids = [model.actuator(name).id for name in act_names]

    # 5) Bucle principal
    while viewer.is_running():
        with viewer.lock():
            t = data.time - STEP_START
            if t < 0:
                data.ctrl[:] = 0
            else:

                # Caderas: senoidal desfasada 120°
                for i, phase in enumerate(PHASES):
                    idx = act_ids[2*i]                # hip_servo
                    data.ctrl[idx] = A_HIP * math.sin(omega*t + phase)


                # Rodillas: rango [0,45] con coseno normalizado
                for i, phase in enumerate(PHASES):
                    idx = act_ids[2*i+1]              # knee_servo
                    # (1+cos)/2 va de 0→1, lo escalamos a 0→A_KNEE
                    data.ctrl[idx] = A_KNEE * (1 + math.cos(omega*t + phase)) / 2

            # Avanza la simulación
            mujoco.mj_step(model, data)

        # Refresca la ventana
        viewer.sync()
        time.sleep(model.opt.timestep * SLOWDOWN)