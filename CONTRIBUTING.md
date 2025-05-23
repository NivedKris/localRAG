# Contributing to HR Policy Assistant

Thank you for considering contributing to the HR Policy Assistant! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming environment for all contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

1. A clear, descriptive title
2. A detailed description of the issue
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Screenshots (if applicable)
7. Environment information (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancements! When suggesting an enhancement:

1. Use a clear and descriptive title
2. Provide a detailed description of the proposed enhancement
3. Explain why this enhancement would be useful
4. Provide examples of how this enhancement would be used

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature` or `git checkout -b fix/your-bugfix`)
3. Make your changes
4. Add tests for your changes if applicable
5. Run tests to ensure they pass
6. Update documentation to reflect your changes
7. Commit your changes with clear, descriptive commit messages
8. Push your branch to your fork
9. Submit a pull request to the main repository

## Development Guidelines

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for Python code
- Use docstrings for all public functions, classes, and methods
- Keep line length to a maximum of 88 characters
- Use 4 spaces for indentation (not tabs)
- Use meaningful variable and function names

### Documentation

- Update the README.md if you change functionality
- Add comments for complex code sections
- Document any new features in appropriate sections

### Testing

- Add tests for new features when possible
- Ensure all tests pass before submitting a pull request
- Test your changes with different types of policy documents

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hr-policy-assistant.git
   cd hr-policy-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Weaviate (using Docker):
   ```bash
   docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.24.8
   ```

5. Install Ollama and required models:
   ```bash
   # Install Ollama from https://ollama.com
   ollama pull all-minilm
   ollama pull tinyllama
   ```

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## License

By contributing to HR Policy Assistant, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

Thank you for contributing to make HR Policy Assistant better!
