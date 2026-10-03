# Releasing

For maintainers. A release is a version number, a tag and a GitHub release; the rest is
automatic.

## What happens on its own

| Event | Result |
|---|---|
| A commit is pushed to `main` | GitHub Actions runs lint, tests, the documentation build and the package build. Read the Docs rebuilds the `latest` documentation |
| A pull request is opened | The same checks, and a preview of the documentation if enabled on Read the Docs |
| A GitHub release is published | The package is built and uploaded to PyPI. Read the Docs builds the `stable` documentation. Zenodo archives the release and gives it a DOI |

A package version on PyPI cannot be replaced, so the package is published per release
and not per commit. The documentation is the part that follows every commit.

## Making a release

1. Add the changes under a heading for the new version in `CHANGELOG.md`.
2. Run the release script with the version:

    ```bash
    scripts/release.sh 0.2.0
    ```

    It checks that the working tree is clean, runs the linter and the unit tests,
    writes the version in `src/spicefault/__init__.py` and `CITATION.cff`, builds the
    package, checks it, commits and creates the tag `v0.2.0`. With `--dry-run` it
    stops before the commit.

3. Push, and publish the release on GitHub:

    ```bash
    git push origin main --follow-tags
    gh release create v0.2.0 --title "spicefault 0.2.0" --notes-from-tag
    ```

To try the upload first, run the *Publish* workflow by hand from the Actions tab: it
uploads the current version to TestPyPI.

## Setting up the services, once

**PyPI.** On [pypi.org](https://pypi.org/manage/account/publishing/) add a pending
trusted publisher: owner `telmomm`, repository `spicefault`, workflow `publish.yml`,
environment `pypi`. Do the same on test.pypi.org with environment `testpypi`. No token
is stored anywhere. In the repository settings, create the two environments.

**Read the Docs.** Import the repository on
[readthedocs.org](https://readthedocs.org/dashboard/import/). The build is configured
by `.readthedocs.yaml`.

**Zenodo.** On [zenodo.org](https://zenodo.org/account/settings/github/) enable the
repository. From the next GitHub release on, each one gets a DOI; the concept DOI,
which always points to the latest version, goes in the badge of the README and in
`CITATION.cff`.

**SonarCloud.** Create the project on [sonarcloud.io](https://sonarcloud.io), set its
keys in `sonar-project.properties`, and add the `SONAR_TOKEN` secret to the repository.
Turn off its automatic analysis, since the analysis runs from CI with coverage. Without
the secret the job is skipped.

**Binder.** Nothing to set up: the `binder/` folder tells mybinder.org how to build
the environment, with ngspice.
