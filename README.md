# AVT-ECoClass-VR-dataset
[![DOI](https://zenodo.org/badge/881347553.svg)](https://doi.org/10.5281/zenodo.14019040)

This is a repository related to the AVT-ECoClass-VR-dataset that is published in course of the paper submitted to the Frontiers research topic "Crossing Sensory Boundaries: Multisensory Perception Through the Lens of Audition".
It contains the data collected during the three subjective experiments described in the mentioned paper.
This work is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project ECoClass-VR (DFG-444697733).

If you make use any of the dataset please cite the following paper:

```bibtex
@article{fremerey2024speakerstory,
  title={Speaker-Story Mapping as a Method to Evaluate Audiovisual Scene Analysis in a Virtual Classroom Scenario},
  author={Stephan Fremerey and Carolin Breuer and Larissa Leist and Maria Klatte and Janina Fels and Alexander Raake},
  journal={Frontiers in Neuroscience},
  year={2024},
  publisher={Frontiers Media SA}
}
```

## Description
The dataset is part of the project [ECoClass-VR](https://www.tu-ilmenau.de/en/university/departments/department-of-electrical-engineering-and-information-technology/profile/institutes-and-groups/audiovisual-technology-group/research/dfg-projekt-ecoclass-vr) to assess how complex visual and acoustic scenes affect cognitive performance in classroom scenarios, across age groups from children to adults.
If you want to download both IVEs and the 360° video and CGI dataset used during this study, please refer to [AVT-ECoClass-VR](https://github.com/Telecommunication-Telemedia-Assessment/AVT-ECoClass-VR).

This study explores how audiovisual Immersive Virtual Environments (IVEs) can assess cognitive performance in classroom-like settings, addressing limitations in simpler acoustic and visual representations.
This paper examines the potential of a test paradigm using speaker-story mapping, called ["audio-visual scene analysis (AV-SA)"](https://backend.orbit.dtu.dk/ws/portalfiles/portal/192631885/001189.pdf), originally developed for Virtual Reality (VR) hearing research, as a method to evaluate audiovisual scene analysis in a virtual classroom scenario.
Factors of acoustic and visual scene representation were varied to investigate their impact on audiovisual scene analysis.
Two acoustic representations were used, a simple diotic, also called monaural, presentation with the same signal presented to both ears ("diotic"), as well as a dynamically live-rendered binaural synthesis ("binaural").
Two visual representations were used: 360° / omnidirectional video with intrinsic lip-sync, and computer-generated imagery (CGI) without lip-sync.
Three subjective experiments were conducted, with different combinations of the two acoustic and visual conditions: The first experiment, involving 36 participants, used 360° video but with "binaural" audio.
The second experiment, with 24 participants, combined 360° video with "diotic" audio.
The third experiment, with 34 participants, used the CGI environment with "binaural" audio.
Each environment presented 20 different speakers in a classroom-like circle of 20 chairs, with the number of simultaneously active speakers ranging from two to ten, while the remaining speakers kept silent and were always shown.
During the experiments, the subjects' task was to correctly map the stories' topics to the corresponding talkers.
The primary dependent variable was the number of correct assignments during a fixed period of 2 min, followed by two questionnaires on mental load after each trial.
In addition, before and / or after the experiments, subjects needed to complete questionnaires about simulator sickness, noise sensitivity, and presence.
Results indicate that the test modality significantly influenced task performance, mental load, and user behaviour but did not affect perceived simulator sickness and presence.
Performance decreased when comparing the 360° video and "binaural" audio experiment with either the experiment using "diotic" audio and 360° , or using "binaural" audio with CGI-based, showing the usefulness of the test method to investigate uni-modal and cross-modal influences on cognitive audiovisual scene analysis performance.

## Structure
* `avrateNG`: Contains a modified version of the [AVrateNG](https://github.com/Telecommunication-Telemedia-Assessment/avrateNG) software. Was used to obtain SSQ questionnaire ratings before and after each experiment, Weinstein's noise sensitivity scale ratings before each experiment, IPQ ratings after each experiment and NASA RTLX and listening effort questionnaire ratings after each trial for each experiment.
* `subjective_data`: Generated data per subjective test
    * `360_binaural`: Generated data for the 360° IVE (binaural audio condition)
        * `avrateNG`: Contains the data recorded with the avrateNG software (SSQ questionnaire ratings, Weinstein's noise sensitivity scale ratings before, IPQ ratings, NASA RTLX ratings, listening effort ratings), as well in native DB as in exported CSV format.
        * `Unity`: Contains the data recorded with the Unity IVE (head and controller rotation values with timestamps, speaker-story mappings) in CSV and JSON format.
    * `360_diotic`: Generated data for the 360° IVE (diotic audio condition)
        * `avrateNG`: Contains the data recorded with the avrateNG software (SSQ questionnaire ratings, Weinstein's noise sensitivity scale ratings before, IPQ ratings, NASA RTLX ratings, listening effort ratings), as well in native DB as in exported CSV format.
        * `Unity`: Contains the data recorded with the Unity IVE (head and controller rotation values with timestamps, speaker-story mappings) in CSV and JSON format.
    * `cgi_binaural`: Generated data for the CGI IVE (binaural audio condition)
        * `avrateNG`: Contains the data recorded with the avrateNG software (SSQ questionnaire ratings, Weinstein's noise sensitivity scale ratings before, IPQ ratings, NASA RTLX ratings, listening effort ratings), as well in native DB as in exported CSV format.
        * `Unity`: Contains the data recorded with the Unity IVE (head and controller rotation values with timestamps, speaker-story mappings) in CSV and JSON format.

*Very important*: For the speaker-story mapping JSON files, note that due to technical reasons the assignments are always stored per file in the `chairs_speaker_story_mapping[Scenenumber-1]` array, hence for example for the file `ecoclass-vr_chairs_speaker_story_mapping_1_Scene4.json` the correct assignments are stored in `chairs_speaker_story_mapping[3]`. The other speaker-story mappings saved in the respective file are invalid, respectively empty. An "assigned_story" of 11 refers to no story.

## License
The contents of the database follow the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

The dataset is part of a publication at Frontiers (see above).