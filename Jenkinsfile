#!groovy
library('sima-jenkins-lib')

def main() {
  def job_name = env.JOB_NAME.split('/')[1]
  def currentBranchName = env.CHANGE_ID ? env.CHANGE_BRANCH : env.BRANCH_NAME

  properties([
      parameters([
          string(name: "COPY_MLA_BRANCH_PKG", defaultValue: currentBranchName, description: 'Copy specified mla pkg'),
          string(name: "COPY_N2A_COMPILER_BRANCH_PKG", defaultValue: currentBranchName, description: 'Copy specified n2a_compiler pkg')
      ]),
  ])

  node("docker") {
    stage("Checkout") {
      utils.checkoutBitbucket()
    }

    def image
    stage("DockerBuild") {
      image = utils.dockerBuild("docker/Dockerfile", 'simaai/' + job_name, "docker_creds", "docker_build.log", { ->
        utils.getPackage('sima-ai','mla', params.COPY_MLA_BRANCH_PKG, '*.deb')
        utils.getPackage('sima-ai','n2a_compiler', params.COPY_N2A_COMPILER_BRANCH_PKG, '*.whl')
        sh "ls -alh"
      })
    }

    parallel push: {
      stage("DockerPush") {
        image['post']()
      }
    }, build: {
      image["image"].inside("-m 32g -c 8") {
        utils.cmakeBuild("build", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", {}, { src_dir ->
          stage("Python Bindings") {
            sh """#!/bin/bash -ex
cd ..
make cython
cd python
python3 setup.py bdist_wheel
"""
          }
        }, "../sima-regres.cmake", "clean all")
        stage("Package") {
          archiveArtifacts('python/dist/*.whl')
          utils.uploadPythonPackages('jenkins_user', 'sima-pypi', 'python/dist/*.whl', 3)
        }
      }
    }
  }
}

utils.job_wrapper( {
  main()
})
