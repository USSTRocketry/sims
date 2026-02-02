from datetime import datetime
from time import process_time, perf_counter, time
import glob
from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
import numpy as np
from numpy.random import normal, uniform, choice
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.figsize"] = [8, 5]
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.titlesize"] = 14

analysis_parameters = {
    # Mass Details
    "rocketMass": (
        15.3,
        0.1,
    ),  # Rocket's dry mass (kg) and its uncertainty (standard deviation)
    # Propulsion Details - run help(SolidMotor) for more information
    "impulse": (9994.5, 0),  # Motor total impulse (N*s)
    "burn_time": (2.92, 0),  # Motor burn out time (s)
    "nozzle_radius": (0.03675, 0.5 / 1000),  # Motor's nozzle radius (m)
    "throat_radius": (.0245, 0.5 / 1000),  # Motor's nozzle throat radius (m)
    "grain_separation": (
        6 / 1000,
        1 / 1000,
    ),  # Motor's grain separation (axial distance between two grains) (m)
    "grainDensity": (1197.82, 50),  # Motor's grain density (kg/m^3)
    "grainOuterRadius": (.049, 0.375 / 1000),  # Motor's grain outer radius (m)
    "grainInitialInnerRadius": (
        .0245,
        0,
    ),  # Motor's grain inner radius (m)
    "grainInitialHeight": (.702, 1 / 1000),  # Motor's grain height (m)
    # Aerodynamic Details - run help(Rocket) for more information
    "inertiaI": (
        .052,
        0.03675,
    ),  # Rocket's inertia moment perpendicular to its axis (kg*m^2)
    "inertiaZ": (
        11.387,
        0.00007,
    ),  # Rocket's inertia moment relative to its axis (kg*m^2)
    "radius": (.065405, 0.001),  # Rocket's radius (kg*m^2)
    "distanceRocketNozzle": (
        -.351,
        0.001,
    ),  # Distance between rocket's center of dry mass and nozzle exit plane (m) (negative)
    "distanceRocketPropellant": (
        0,
        0.001,
    ),  # Distance between rocket's center of dry mass and and center of propellant mass (m) (negative)
    "powerOffDrag": (
        0.9081 / 1.05,
        0.033,
    ),  # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "powerOnDrag": (
        0.9081 / 1.05,
        0.033,
    ),  # Multiplier for rocket's drag curve. Usually has a mean value of 1 and a uncertainty of 5% to 10%
    "noseLength": (0.7239, 0.001),  # Rocket's nose cone length (m)
    "noseDistanceToCM": (
        .80,
        0.001,
    ),  # Axial distance between rocket's center of dry mass and nearest point in its nose cone (m)
    "finSpan": (0.1397, 0),  # Fin span (m)
    "finRootChord": (0.2286, 0.0005),  # Fin root chord (m)
    "finTipChord": (0.0508, 0.0005),  # Fin tip chord (m)
    "finDistanceToCM": (
        .7112,
        0.001,
    ),  # Axial distance between rocket's center of dry mass and nearest point in its fin (m)
    # Launch and Environment Details - run help(Environment) and help(Flight) for more information
    "inclination": (
        89,
        1,
    ),  # Launch rail inclination angle relative to the horizontal plane (degrees)
    "heading": (53, 2),  # Launch rail heading relative to north (degrees)
    "railLength": (4.97, 0.0005),  # Launch rail length (m)
    # Parachute Details - run help(Rocket) for more information
    "CdSDrogue": (
        0.2918635 * 1.55,
        0.07,
    ),  # Drag coefficient times reference area for the drogue chute (m^2)
    "lag_rec": (
        1.5,
        0.5,
    ),  # Time delay between parachute ejection signal is detected and parachute is inflated (s)
    # Electronic Systems Details - run help(Rocket) for more information
    "lag_se": (
        0.73,
        0.16,
    ),  # Time delay between sensor signal is received and ejection signal is fired (s)
}


def flight_settings(analysis_parameters, total_number):
    i = 0
    while i < total_number:
        # Generate a flight setting
        flight_setting = {}
        for parameter_key, parameter_value in analysis_parameters.items():
            if type(parameter_value) is tuple:
                flight_setting[parameter_key] = normal(*parameter_value)
            else:
                flight_setting[parameter_key] = choice(parameter_value)

        # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
        if flight_setting["lag_rec"] < 0 or flight_setting["lag_se"] < 0:
            continue

        # Update counter
        i += 1
        # Yield a flight setting
        yield flight_setting


def export_flight_data(flight_setting, flight_data, exec_time):
    # Generate flight results
    flight_result = {
        "outOfRailTime": flight_data.out_of_rail_time,  # FIXED: snake_case
        "outOfRailVelocity": flight_data.out_of_rail_velocity,  # FIXED: snake_case
        "apogeeTime": flight_data.apogee_time,  # FIXED: snake_case
        "apogeeAltitude": flight_data.apogee - Env.elevation,
        "apogeeX": flight_data.apogee_x,  # FIXED: snake_case
        "apogeeY": flight_data.apogee_y,  # FIXED: snake_case
        "impactTime": flight_data.t_final,  # FIXED: snake_case
        "impactX": flight_data.x_impact,  # FIXED: snake_case
        "impactY": flight_data.y_impact,  # FIXED: snake_case
        "impactVelocity": flight_data.impact_velocity,  # FIXED: snake_case
        "initialStaticMargin": flight_data.rocket.static_margin(0),  # FIXED: snake_case
        "outOfRailStaticMargin": flight_data.rocket.static_margin(
            flight_data.out_of_rail_time  # FIXED: Use flight_data, not TestFlight
        ),
        "finalStaticMargin": flight_data.rocket.static_margin(
            flight_data.rocket.motor.burn_out_time  # FIXED: snake_case
        ),
        "numberOfEvents": len(flight_data.parachute_events),  # FIXED: snake_case
        "executionTime": exec_time,
    }

    # Calculate maximum reached velocity
    sol = np.array(flight_data.solution)
    flight_data.vx = Function(
        sol[:, [0, 4]], "Time (s)", "Vx (m/s)", "linear", extrapolation="natural"
    )
    flight_data.vy = Function(
        sol[:, [0, 5]], "Time (s)", "Vy (m/s)", "linear", extrapolation="natural"
    )
    flight_data.vz = Function(
        sol[:, [0, 6]], "Time (s)", "Vz (m/s)", "linear", extrapolation="natural"
    )
    flight_data.v = (
                            flight_data.vx ** 2 + flight_data.vy ** 2 + flight_data.vz ** 2
                    ) ** 0.5
    flight_data.maxVel = np.amax(flight_data.v.source[:, 1])
    flight_result["maxVelocity"] = flight_data.maxVel

    # Take care of parachute results
    if len(flight_data.parachute_events) > 0:  # FIXED: snake_case
        flight_result["drogueTriggerTime"] = flight_data.parachute_events[0][0]
        flight_result["drogueInflatedTime"] = (
                flight_data.parachute_events[0][0] + flight_data.parachute_events[0][1].lag
        )
        flight_result["drogueInflatedVelocity"] = flight_data.v(
            flight_data.parachute_events[0][0] + flight_data.parachute_events[0][1].lag
        )
    else:
        flight_result["drogueTriggerTime"] = 0
        flight_result["drogueInflatedTime"] = 0
        flight_result["drogueInflatedVelocity"] = 0

    # Write flight setting and results to file
    dispersion_input_file.write(str(flight_setting) + "\n")
    dispersion_output_file.write(str(flight_result) + "\n")


def export_flight_error(flight_setting):
    dispersion_error_file.write(str(flight_setting) + "\n")


# -------------------------------------------------------------------------------
# Basic analysis info
filename = "C:/Users/hawai/PycharmProjects/EverythingBagel2/up2_run1"
number_of_simulations = 2

# Create data files for inputs, outputs and error logging
dispersion_error_file = open(str(filename) + ".disp_errors.txt", "w")
dispersion_input_file = open(str(filename) + ".disp_inputs.txt", "w")
dispersion_output_file = open(str(filename) + ".disp_outputs.txt", "w")

# Initialize counter and timer
i = 0

initial_wall_time = time()
initial_cpu_time = process_time()

# Define basic Environment object
Env = Environment(
    date=(2025, 8, 10, 10),
    latitude=47.98694,
    longitude=-81.84829
)
Env.set_elevation(295)
Env.max_expected_height = 5000
Env.set_atmospheric_model(type="standard_atmosphere")


def drogueTrigger(p, h, y):
    # Check if rocket is going down, i.e. if it has passed the apogee
    vertical_velocity = y[5]
    # Return true to activate parachute once the vertical velocity is negative
    return True if vertical_velocity < 0 else False


# Iterate over flight settings
print("Starting")
for setting in flight_settings(analysis_parameters, number_of_simulations):
    start_time = process_time()
    i += 1

    # Create motor
    Cesaroni = SolidMotor(
        thrust_source="C:/Users/hawai/PycharmProjects/EverythingBagel2/thrust_source.csv",
        dry_mass=15.3,
        dry_inertia=(0.052, 0.052, 11.387),
        burn_time=2.92,
        reshape_thrust_curve=(setting["burn_time"], setting["impulse"]),
        nozzle_radius=setting["nozzle_radius"],
        throat_radius=setting["throat_radius"],
        grains_center_of_mass_position=setting["distanceRocketPropellant"],
        grain_number=4,
        grain_separation=setting["grain_separation"],
        grain_density=setting["grainDensity"],
        grain_outer_radius=setting["grainOuterRadius"],
        grain_initial_inner_radius=setting["grainInitialInnerRadius"],
        grain_initial_height=setting["grainInitialHeight"],
        interpolation_method="linear",
        center_of_dry_mass_position=0,
        nozzle_position=setting["distanceRocketNozzle"],
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    # Create rocket - FIXED: All parameters to snake_case and proper values
    Valetudo = Rocket(
        radius=setting["radius"],
        mass=setting["rocketMass"],
        inertia=(setting["inertiaI"], setting["inertiaI"], setting["inertiaZ"]),  # FIXED: Use setting values
        power_off_drag="C:/Users/hawai/PycharmProjects/EverythingBagel2/drag_curve.csv",
        # FIXED: Need file path
        power_on_drag="C:/Users/hawai/PycharmProjects/EverythingBagel2/drag_curve.csv",
        # FIXED: Need file path
        center_of_mass_without_motor=0,  # FIXED: Changed from 1 to 0 (typical default)
        coordinate_system_orientation="tail_to_nose",
    )

    # FIXED: set_rail_buttons (snake_case)
    Valetudo.set_rail_buttons(
        upper_button_position=0.224,
        lower_button_position=-0.93,
        angular_position=30
    )

    # FIXED: add_motor (snake_case)
    Valetudo.add_motor(Cesaroni, position=setting["distanceRocketNozzle"])

    # Edit rocket drag
    Valetudo.power_off_drag *= setting["powerOffDrag"]  # FIXED: snake_case
    Valetudo.power_on_drag *= setting["powerOnDrag"]  # FIXED: snake_case

    # FIXED: add_nose (snake_case)
    NoseCone = Valetudo.add_nose(
        length=setting["noseLength"],
        kind="vonkarman",  # FIXED: snake_case
        position=setting["noseDistanceToCM"] + setting["noseLength"],
    )

    # FIXED: add_trapezoidal_fins (snake_case)
    FinSet = Valetudo.add_trapezoidal_fins(
        n=3,
        span=setting["finSpan"],
        root_chord=setting["finRootChord"],  # FIXED: snake_case
        tip_chord=setting["finTipChord"],  # FIXED: snake_case
        position=setting["finDistanceToCM"],
        cant_angle=0,  # FIXED: snake_case
        airfoil=None,
    )

    # FIXED: add_parachute (snake_case)
    Drogue = Valetudo.add_parachute(
        "Drogue",
        cd_s=setting["CdSDrogue"],  # FIXED: snake_case
        trigger=drogueTrigger,
        sampling_rate=105,  # FIXED: snake_case
        lag=setting["lag_rec"] + setting["lag_se"],
        noise=(0, 8.3, 0.5),
    )

    # Run trajectory simulation
    try:
        TestFlight = Flight(
            rocket=Valetudo,
            environment=Env,
            rail_length=setting["railLength"],  # FIXED: snake_case
            inclination=setting["inclination"],
            heading=setting["heading"],
            max_time=600,  # FIXED: snake_case
        )
        export_flight_data(setting, TestFlight, process_time() - start_time)
    except Exception as E:
        print(f"Error in iteration {i}: {E}")
        export_flight_error(setting)

    # Register time
    print(f"Current iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time) / i:2.6f} s")

# Done

## Print and save total time
final_string = f"Completed {i} iterations successfully. Total CPU time: {process_time() - initial_cpu_time} s. Total wall time: {time() - initial_wall_time} s"
print(final_string)
dispersion_input_file.write(final_string + "\n")
dispersion_output_file.write(final_string + "\n")
dispersion_error_file.write(final_string + "\n")

## Close files
dispersion_input_file.close()
dispersion_output_file.close()
dispersion_error_file.close()

filename = "C:/Users/hawai/PycharmProjects/EverythingBagel2/dispersion_analysis_outputs/up2_run1"  # FIXED: Full path

# Initialize variable to store all results
dispersion_general_results = []

dispersion_results = {
    "outOfRailTime": [],
    "outOfRailVelocity": [],
    "apogeeTime": [],
    "apogeeAltitude": [],
    "apogeeX": [],
    "apogeeY": [],
    "impactTime": [],
    "impactX": [],
    "impactY": [],
    "impactVelocity": [],
    "initialStaticMargin": [],
    "outOfRailStaticMargin": [],
    "finalStaticMargin": [],
    "numberOfEvents": [],
    "maxVelocity": [],
    "drogueTriggerTime": [],
    "drogueInflatedTime": [],
    "drogueInflatedVelocity": [],
    "executionTime": [],
}

# Get all dispersion results
# Get file
try:
    dispersion_output_file = open(str(filename) + ".disp_outputs.txt", "r+")

    # Read each line of the file and convert to dict
    for line in dispersion_output_file:
        # Skip comments lines
        if line[0] != "{":
            continue
        # Eval results and store them
        flight_result = eval(line)
        dispersion_general_results.append(flight_result)
        for parameter_key, parameter_value in flight_result.items():
            dispersion_results[parameter_key].append(parameter_value)

    # Close data file
    dispersion_output_file.close()

    # Print number of flights simulated
    N = len(dispersion_general_results)
    print("Number of simulations: ", N)

    # Only plot if we have results
    if N > 0:
        # Import libraries
        from imageio import imread
        from matplotlib.patches import Ellipse

        # Import background map
        img = imread("C:/Users/hawai/PycharmProjects/EverythingBagel2/Downloads/basemap_one.png")  # FIXED: Full path

        # Retrieve dispersion data por apogee and impact XY position
        apogeeX = np.array(dispersion_results["apogeeX"])
        apogeeY = np.array(dispersion_results["apogeeY"])
        impactX = np.array(dispersion_results["impactX"])
        impactY = np.array(dispersion_results["impactY"])


        # Define function to calculate eigen values
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]


        # Create plot figure
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor="w", edgecolor="k")
        ax = plt.subplot(111)

        # Calculate error ellipses for impact
        impactCov = np.cov(impactX, impactY)
        impactVals, impactVecs = eigsorted(impactCov)
        impactTheta = np.degrees(np.arctan2(*impactVecs[:, 0][::-1]))
        impactW, impactH = 2 * np.sqrt(impactVals)

        # Draw error ellipses for impact
        impact_ellipses = []
        for j in [1, 2, 3]:
            impactEll = Ellipse(
                xy=(np.mean(impactX), np.mean(impactY)),
                width=impactW * j,
                height=impactH * j,
                angle=impactTheta,
                color="black",
            )
            impactEll.set_facecolor((0, 0, 1, 0.2))
            impact_ellipses.append(impactEll)
            ax.add_artist(impactEll)

        # Calculate error ellipses for apogee
        apogeeCov = np.cov(apogeeX, apogeeY)
        apogeeVals, apogeeVecs = eigsorted(apogeeCov)
        apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:, 0][::-1]))
        apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)

        # Draw error ellipses for apogee
        for j in [1, 2, 3]:
            apogeeEll = Ellipse(
                xy=(np.mean(apogeeX), np.mean(apogeeY)),
                width=apogeeW * j,
                height=apogeeH * j,
                angle=apogeeTheta,
                color="black",
            )
            apogeeEll.set_facecolor((0, 1, 0, 0.2))
            ax.add_artist(apogeeEll)

        # Draw launch point
        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        # Draw apogee points
        plt.scatter(apogeeX, apogeeY, s=5, marker="^", color="green", label="Simulated Apogee")
        # Draw impact points
        plt.scatter(
            impactX, impactY, s=5, marker="v", color="blue", label="Simulated Landing Point"
        )
        # Draw real landing point
        plt.scatter(
            411.89, -61.07, s=20, marker="X", color="red", label="Measured Landing Point"
        )

        plt.legend()

        # Add title and labels to plot
        ax.set_title("Apogee and Landing Points")
        ax.set_ylabel("North (m)")
        ax.set_xlabel("East (m)")

        # Add background image to plot
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        plt.imshow(img, zorder=0, extent=[-1000 - dx, 1000 - dx, -1000 - dy, 1000 - dy])
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(-100, 700)
        plt.ylim(-300, 300)

        # Save plot and show result
        plt.savefig(str(filename) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(str(filename) + ".svg", bbox_inches="tight", pad_inches=0)
        plt.show()
    else:
        print("No simulation results to plot.")

except FileNotFoundError:
    print(f"Output file not found: {filename}.disp_outputs.txt")
    print("Skipping plotting. Make sure simulations completed successfully.")