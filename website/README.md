# COMP5434 Project Dashboard

This is a static website created to present the project roadmap and performance metrics for non-technical team members.

## How to serve locally

To view the dashboard with the charts correctly loading the JSON data, run the following command in the project root directory:

```bash
python -m http.server 8000 --directory website/
```

Then open your browser and navigate to `http://localhost:8000/`.

## Deployment

This website can be easily deployed using **GitHub Pages**:
1. Go to your repository settings on GitHub.
2. Navigate to "Pages" under the "Code and automation" section.
3. Select the branch you want to deploy from.
4. Select the `/website` folder as the source (or deploy from root if moved).
5. Save, and your site will be live!