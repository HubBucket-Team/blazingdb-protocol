from setuptools import setup


setup(name='blazingdb-protocol',
      version='1.0',
      description='Messaging system for BlazingDB',
      author='BlazingDB Team',
      author_email='blazing@blazingdb',
      url='https://github.com/BlazingDB/blazingdb-protocol',
      packages=['blazingdb.protocol'],
      install_requires=['flatbuffers'],
)
