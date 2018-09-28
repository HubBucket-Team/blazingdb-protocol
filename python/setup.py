from subprocess import check_call

from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info as install


class InstallCommand(install):

  def run(self):
    check_call(('sh', 'resources/generate'))
    install.run(self)


setup(name='blazingdb-protocol',
      version='1.0',
      description='Messaging system for BlazingDB',
      author='BlazingDB Team',
      author_email='blazing@blazingdb',
      url='https://github.com/BlazingDB/blazingdb-protocol',
      packages=find_packages(),
      install_requires=['flatbuffers'],
      cmdclass={'egg_info': InstallCommand},
)
