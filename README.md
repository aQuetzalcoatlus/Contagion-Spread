# Contagion Agent Simulation

<p align="center">
  <img src="./resources/Contagion_repo_image_resized.png" width="500"/>
</p>

This project is an interactive Streamlit app that demonstrates how an infection can spread through a simple population of moving particles. Each particle behaves like an agent that can be healthy, infected, or immunized. By adjusting parameters such as vaccination rate, breakthrough probability, and infection duration, users can explore how different conditions shape the development of an outbreak. The app builds on the agent simulation task completed as part of the Classical Complex Systems course I completed during the winter semester 2023/24 at the University of Freiburg.

## Features

- Agent based simulation with elastic hard sphere collisions  
- Three health states: healthy, infected, immunized  
- Infection transfer on collision, including configurable breakthrough infections  
- Recovery after a fixed number of steps  
- Single simulation view with metrics and infection curves  
- Comparison of multiple vaccination scenarios  
- Interactive controls for population size, step count, immunization percentage, breakthrough probability, and more  

## Running the app

You can run the Streamlit interface with:

```bash
uv run streamlit run src/contagion_app/app.py
```

If you are using a virtual environment created by `uv`, make sure the dependencies listed in `pyproject.toml` are installed through:

```bash
uv sync
```

## Purpose

This project is meant for experimentation and learning, completed as part of the  Classical Complex Systems course at the University of Freiburg. It helps users see how vaccination levels affect the number of infected agents over time and whether an infection dies out or becomes endemic. 