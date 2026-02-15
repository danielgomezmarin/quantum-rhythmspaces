# quantum-rhythmspaces

This is the code repository for the paper *From Qubits to Rhythm: Exploring Quantum Random Walks in Rhythmspaces*, which presents a proof-of-concept framework for generating drum patterns by mapping 2D quantum random walks to a rhythmspace.

## Files explanation for reproducibility:

### Folders

- **`__pycache__/`**  
  Python cache files.

- **`midi/`**  
  MIDI files generated for percussion/rhythm outputs.

- **`path_txts/`**  
  `.txt` files containing trajectory/path examples generated for the paper.

- **`rhythm-env/`**  
  Local environment folder used during development.

---

### Data/cache files for rhythmspace precomputation

These files were saved to avoid rebuilding the rhythmspace from scratch during repeated testing:

- **`all_names.txt`**
- **`all_pattlists.txt`**
- **`descriptors.txt`**
- **`positions.txt`**

---

### Python scripts

- **`descriptors.py`**  
  Existing script used to handle rhythm descriptors.

- **`examples.py`**  
  Script that generates the rhythmspace.

- **`functions.py`**  
  Base rhythmspace utility functions.  
  Additional functions for quantum walks were appended at the end (with comments indicating the additions).

- **`qufunctions.py`**  
  New functions created specifically for the quantum module/workflow.

- **`generate_path_and_animation.py`**  
  Generates quantum trajectories/paths and their animations.

- **`generate_midi.py`**  
  Generates MIDI output from a quantum path/trajectory.

- **`merge_audio_and_video.py`**  
  Merges rendered audio and video from a trajectory into a final video.

- **`metric_optimizer.py`**  
  Metric/parameter optimization used to run the quantum walk experiments.

- **`install_requirements.py`**  
  Helper script to install dependencies.

---

### Notebook

- **`potentials_subplots.ipynb`**  
  Notebook used to generate potential-field subplot figures for the paper.

---

### Dependencies file

- **`requirements.txt`**  
  Python dependencies for the project.


## Contact

- **María Aguado-Yáñez**  
  Interdisciplinary Centre for Computer Music Research (ICCMR), University of Plymouth  
  📧 maria.aguadoyanez@plymouth.ac.uk


