# VideoChatBot

VideoChatBot is a library that gives python users an interaction utility that works as a videocall to a bot.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## How it works

An istance of vcbot starts with default chat data as its training set. The run function of this instance is used to launch the interaction. Thereafter, two threads are instantiated for video display and ensemble process. The ensemble works by taking user image and audio as input, and generates a response by a standard chatterbot which is fed with user emotion (inferred from user image and audio) along with the user chat text (inferred from audio).

## Installation

This package can be installed from [PyPi](https://pypi.python.org/pypi/VideoChatBot) by running:

```
pip install videochatbot
```

## Basic Usage

```
from VideoChatBot import vcbot

mybot = vcbot()
mybot.run()

```

# History

See release notes for changes https://github.com/avaish1409/VideoChatBot/releases

# Development pattern for contributors

1. [Create a fork](https://help.github.com/articles/fork-a-repo/) of
   the [main VideoChatBot repository](https://github.com/avaish1409/VideoChatBot) on GitHub.
2. Make your changes in a branch named something different from `master` and titled as per your contribution, e.g. create
   a new branch `phonetics-based-gif`.
3. [Create a pull request](https://help.github.com/articles/creating-a-pull-request/).
4. Please follow the [Python style guide for PEP-8](https://www.python.org/dev/peps/pep-0008/).

# License

VideoChatBot is licensed under the [BSD 3-clause license](https://opensource.org/licenses/BSD-3-Clause).
