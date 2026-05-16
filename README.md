# Udacity Agentic AI Nanodegree - Multi-Agent Systems Project

This repository contains the implementation for the Project of **Course 4: Multi-Agent Systems**,
of the **Udacity Agentic AI** Nanodegree program.

## Getting Started

### Dependencies

* **Python 3.14+**
* **Poetry** (for dependency management)
* **API Keys**:
    * OpenAI API Key

To install Poetry, follow the instructions: https://python-poetry.org/docs/

### Installation

**1\. Clone the repository**

**2\. Install dependencies**\
Use Poetry to create a virtual environment and install the required packages.\
From the root project directory, run the following command:
```shell
poetry install
```

**3\. Add a `.env` file** inside the `/project` directory.\
File must have the following variables:
```dotenv
OPENAI_BASE_URL=https://openai.vocareum.com/v1
OPENAI_API_KEY=voc-***.***
```

## Project Structure
`./project`: Contains the implementation and additional files to run the main application
  * `/project/project.py`: Agent implementation file
  * `/project/DIAGRAM.md`: Updated agent flow diagram 

`./starter`: **Untouched** starter template along with Project Overview and Instructions

`./tests`: Unit tests directory

## Running the agent application

The project is implemented in the `.project/project.py` file

To run it using poetry, execute the following command from the root project directory:
```shell
poetry run python project/project.py
```

## Project reflections
[REFLECTIONS.md](./REFLECTIONS.md)

## Built With
* Poetry: Dependency Management
* SQLAlchemy: Database
* Pandas: Data Manipulation
* OpenAI: LLM Provider
* Pydantic: Data Validation
* Pydantic AI: Agent implementation and evaluation

## License
[LICENSE.md](./LICENSE.md)
