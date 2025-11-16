# Contributing to YouTube Views Predictor

Thank you for your interest in contributing to the YouTube Views Predictor project! This document provides guidelines for contributing to the project.

## 🌟 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots if applicable

### Suggesting Features

We welcome feature suggestions! Please create an issue with:
- A clear description of the feature
- Use cases and benefits
- Any implementation ideas you have

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the project structure** outlined in PROJECT_OUTLINE.md
3. **Write clear, documented code** following Python best practices
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Ensure all tests pass** before submitting
7. **Submit a pull request** with a clear description

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused on a single task

### Code Formatting
```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

### Docstring Format
Use Google style docstrings:
```python
def predict_views(title: str, keywords: list) -> int:
    """Predict video views based on metadata.
    
    Args:
        title: Video title
        keywords: List of video keywords
        
    Returns:
        Predicted view count
        
    Raises:
        ValueError: If title is empty
    """
    pass
```

## 🧪 Testing

- Write unit tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage
- Test edge cases and error conditions

## 📚 Documentation

- Update README.md for user-facing changes
- Update PROJECT_OUTLINE.md for structural changes
- Add inline comments for complex logic
- Update API documentation for new endpoints

## 🔄 Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes and commit: `git commit -m "Add feature: description"`
3. Push to your fork: `git push origin feature/your-feature-name`
4. Open a pull request with a clear description

## ✅ Pull Request Checklist

Before submitting a PR, ensure:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear and descriptive

## 🎯 Priority Areas

Current priorities for contributions:
1. Data collection pipeline
2. Feature engineering techniques
3. Model implementations
4. API development
5. Testing coverage
6. Documentation improvements

## 💬 Getting Help

- Open an issue for questions
- Review PROJECT_OUTLINE.md for project structure
- Check existing issues and PRs before starting work

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to YouTube Views Predictor! 🚀
