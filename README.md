# Autopilot-simulation-using-DeepQLearning# Autopilot Simulation Using Deep Q-Learning

## Introduction

This project demonstrates an autonomous vehicle simulation using a Deep Q-Learning Network (DQN) in the CARLA environment. The DQN model is responsible for training the self-driving car to navigate safely and efficiently in a simulated environment.

## Prerequisites

Before running the simulation, ensure you have the following prerequisites installed on your local system:

- CARLA 0.9.13
- Python 3.x
- Git

## Setup

Follow these steps to set up and run the simulation:

1. Download CARLA 0.9.13:
   - Visit the CARLA releases page: [CARLA Releases](https://github.com/carla-simulator/carla/releases)
   - Download and install CARLA 0.9.13 for your platform.

2. Clone this repository:
   - Open a terminal or command prompt.
   - Navigate to the parent folder where you want to place the CARLA example.
   - Change directory to CARLA_0.9.13/PythonAPI/examples:
     ```
     cd CARLA_0.9.13/PythonAPI/examples
     ```
   - Clone this repository:
     ```
     git clone <repository-url>
     ```

3. Install dependencies:
   - Install the required Python packages by running:
     ```
     pip install -r requirements.txt
     ```

4. Run the Simulation:
   - Execute the simulation script:
     ```
     python run.py
     ```


## Usage

Once the setup is complete, you can interact with the autonomous vehicle simulation using the provided script. The trained DQN model will control the vehicle's navigation, demonstrating the capabilities of AI in autonomous driving.

Feel free to explore, modify, and improve the simulation as needed for your research or learning purposes.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code as per the terms of the license.

Happy Autonomous Driving!
