# Cyclops Wide-Field Rig
This is a repository for the Cyclops wide-field calcium imaging rig built by myself and [G. Lopes](https://neurogears.org/) for my experiments conducted at the [Hofer](https://www.sainsburywellcome.org/web/groups/hofer-lab) and [Mrsic-Flogel](https://www.sainsburywellcome.org/web/groups/mrsic-flogel-lab) labs at the Sainsbury-Wellcome Centre.

Wide-field calcium imaging is an exciting approach for functional imaging of the mouse brain whereby activity across the entire dorsal surface of the cerebral cortex can be imaged and observed in real-time with minimal processing, while the head-fixed animal engages in a behavioural task. It is in use by multiple labs around the world to provide insight about mesoscale dynamics and interactions between cortical networks in behaving animals.

The rig was built from scratch from specifications identical to the setups created and developed originally by Ivana Orsolic and Kelly Clancy, as seen in the following publications:
- [Orsolic et al., 2019: Mesoscale cortical dynamics reflect the interaction of sensory evidence and temporal expectation during perceptual decision-making](https://www.biorxiv.org/content/10.1101/552026v1)
- [Clancy et al., 2019: Locomotion-dependent remapping of distributed cortical networks](https://www.nature.com/articles/s41593-019-0357-8)

The main difference for this setup is it is fully run & controlled by [Bonsai software](https://bonsai-rx.org/). Bonsai is a visual programming language designed to control data acquisition & instrument actuation in a fast & responsive way. In this setup, Bonsai controls camera frame acquisition, UV/Blue illumination, keeps track of dropped frames & implements a full 'state machine' via Nidaq boards where the status of running behaviour, eye and body cameras, optogenetic stimulation status & wavelength currently being exposed is fully tracked.

In this repository we share the Bonsai workflows used for for running the setup and Python code used to parse & analyse data it generates.

## Hardware components
Harp: https://www.cf-hw.org/harp
Behavior board: https://www.cf-hw.org/harp/behavior
Drivers: https://bitbucket.org/fchampalimaud/downloads/downloads/
