# Contributing to YouTube Views Predictor

Thank you for your interest in contributing to YouTube Views Predictor! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in your interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YoutubeViewsPredictor.git
   cd YoutubeViewsPredictor
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/shree-crypto/YoutubeViewsPredictor.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Train the initial model**:
   ```bash
   python train_model.py
   ```

## Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Write or update tests** for your changes

4. **Run tests** to ensure everything works:
   ```bash
   pytest test_basic.py
   ```

5. **Update documentation** if needed

## Testing

### Running Tests

Run all tests:
```bash
pytest test_basic.py
```

Run with coverage:
```bash
pytest test_basic.py --cov=utils --cov-report=html
```

### Writing Tests

- Place tests in files matching `test_*.py` or `*_test.py`
- Use descriptive test names: `test_feature_extraction_handles_empty_title`
- Test edge cases and error conditions
- Aim for high code coverage (>80%)

## Code Style

This project follows these style guidelines:

### Python Style

- **PEP 8** compliance (enforced by flake8)
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** for function parameters and returns
- **Docstrings** for all public functions and classes (Google style)

### Example Function

```python
def extract_features(title: str, duration: int) -> Dict[str, Any]:
    """
    Extract features from video metadata.
    
    Args:
        title: Video title
        duration: Duration in seconds
        
    Returns:
        Dictionary of extracted features
        
    Raises:
        ValueError: If title is empty or duration is negative
    """
    if not title:
        raise ValueError("Title cannot be empty")
    
    # Implementation here
    return features
```

### Running Code Quality Tools

```bash
# Format code
black .

# Sort imports
isort .

# Check for issues
flake8 .

# Type checking
mypy utils/
```

## Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: video category analysis"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Reference to any related issues
   - Screenshots for UI changes
   - Test results

4. **Wait for review** and address any feedback

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Areas for Contribution

We welcome contributions in these areas:

### Features
- YouTube API integration for real data
- Deep learning models (BERT, Transformers)
- Advanced visualization dashboards
- A/B testing framework
- Real-time prediction API
- Mobile app interface

### Improvements
- Performance optimization
- Better error handling
- More comprehensive tests
- Improved documentation
- Accessibility enhancements

### Bug Fixes
- Check the [Issues](https://github.com/shree-crypto/YoutubeViewsPredictor/issues) page
- Look for issues tagged `good first issue` or `help wanted`

## Questions?

- Open an [Issue](https://github.com/shree-crypto/YoutubeViewsPredictor/issues)
- Start a [Discussion](https://github.com/shree-crypto/YoutubeViewsPredictor/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
