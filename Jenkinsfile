#!groovy
library('sima-jenkins-lib')

def main() {
  def job_name = env.JOB_NAME.split('/')[1]

  properties([
      parameters([
          booleanParam(
              name: 'SKIP_N2A_COMPILER_BUILD',
              description: 'Skips building n2a_compiler',
              defaultValue: false
          ),
          booleanParam(name: "PACKAGE_ONLY", defaultValue: false, description: 'Only package don\'t run tests')
      ]),
  ])

  node("docker") {
    stage("Checkout") {
      utils.checkoutBitbucket()
    }

    def image
    stage("DockerBuild") {
      image = utils.dockerBuild("docker/Dockerfile", 'simaai/' + job_name, "docker_creds", "docker_build.log", { ->
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
            dir("../python") {
              utils.setPythonBuildEnv([]) {
                sh """#!/bin/bash -ex
rm -rf dist build
python3 setup.py bdist_wheel
"""
              }
            }
          }
        }, "../sima-regres.cmake", "clean all")
        stage("Package") {
          archiveArtifacts('python/dist/*.whl')
          utils.uploadPythonPackages('jenkins_user', 'sima-pypi', 'python/dist/*.whl', 3)
        }
      }
    }

    stage("Promotion") {
      if (env.BRANCH_NAME=="sima") {
        utils.docker_promote(image['image'], 'docker_creds', '')
      }
    }

  }

  stage("Upstream") {
    utils.buildUpstream("n2a_compiler", params.SKIP_N2A_COMPILER_BUILD, [
        booleanParam(name: 'PACKAGE_ONLY', value: params.PACKAGE_ONLY)
    ])
  }
}

utils.job_wrapper( {
  main()
})

return this
