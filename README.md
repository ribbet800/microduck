# Microduck

Simulation and reinforcement-learning work for a Microduck bipedal robot: building a
simulated model, training locomotion policies, and getting them ready to transfer to the
real robot when it arrives.

## Status

Early setup — nothing to run yet.

## Plan

- [ ] Set up the robot model (URDF/MJCF) and a physics simulator
- [ ] Get the model standing and walking under scripted control in sim
- [ ] Train a locomotion policy with RL
- [ ] Add domain randomisation for sim-to-real robustness
- [ ] Deploy and test on the physical robot

## Layout

```
microduck/
├── README.md
├── LICENSE
└── .gitignore
```

## License

MIT — see [LICENSE](LICENSE).
