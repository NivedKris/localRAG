# Instructions for Pushing to GitHub

## 1. Create a New Repository on GitHub

1. Go to https://github.com/
2. Log in to your GitHub account
3. Click on the "+" icon in the top right corner
4. Select "New repository"
5. Enter a repository name (e.g., "local-rag-system")
6. Optionally add a description
7. Choose whether to make the repository public or private
8. Do NOT initialize the repository with a README, .gitignore, or license
9. Click "Create repository"

## 2. Push Your Local Repository to GitHub

After creating the repository, GitHub will show you commands to push an existing repository. Run the following commands in your terminal:

```bash
# Replace YOUR_USERNAME with your GitHub username and REPO_NAME with your repository name
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main  # Rename master branch to main (recommended)
git push -u origin main
```

If you're using SSH instead of HTTPS:

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## 3. Authentication

- If using HTTPS, you'll be prompted to enter your GitHub username and password
- GitHub no longer accepts passwords for Git operations, so you'll need to use a personal access token instead of your password:
  - Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Generate New Token
  - Check the "repo" scope and any other permissions you need
  - Generate the token and use it as your password when prompted

- Alternatively, use the GitHub CLI to authenticate:
  ```bash
  # Install GitHub CLI if not already installed
  # Then authenticate
  gh auth login
  ```

## 4. Verify the Push

After pushing, refresh your GitHub repository page to see your files.
