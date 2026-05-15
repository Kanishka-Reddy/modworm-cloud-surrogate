"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import os
import platform

platform = platform.system()
default_dir = os.getcwd()

main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, 'data')
muscle_maps_dir = os.path.join(main_dir, 'muscle_maps')
presets_input_dir = os.path.join(main_dir, 'presets_input')
presets_voltage_dir = os.path.join(main_dir, 'presets_voltage')
videos_dir = os.path.join(default_dir, 'created_vids')