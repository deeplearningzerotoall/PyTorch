# ‘모두가 만드는 모두를 위한 딥러닝’ 참여 방법!! (Contribution)

## Precheck steps : 사전 확인

* 작업을 시작하기 전에 먼저 이슈를 남겨 두세요. 왜냐면
  * 여러분이 무엇을 하고 있는지 사람들에게 알리는 데 도움이 됩니다.
  * 제안하는 문제가 이 Repo와 무관할 수 있습니다.
  * 그런 방식으로 코드를 유지하는 게 우리의 의도일 수도 있습니다. ([KISS](https://en.wikipedia.org/wiki/KISS_principle))
* 여러분은 Git을 어떻게 사용하는지 알아야합니다.
  * 그렇지 않다면, "Git 사용 방법"을 검색한 후, 무언가를 하기 전에 그것들을 읽어 보세요. 개발자로서 살아남기 위해서는 필수적인 기술입니다.
  * [Git tutorial](https://try.github.io/levels/1/challenges/1)을 참고하세요. 

## Contribution guidelines

이 문서는 Contribution 프로세스를 안내합니다.

### Step 1: Fork

Fork 버튼을 눌러 [GitHub](https://github.com/deeplearningzerotoall/PyTorch.git)에 프로젝트를 Fork하세요. 이 단계는 작업을 시작할 수 있게 여러분의 계정에 복사하게 됩니다.

### Step 2: Local computer에 다운로드하세요

```bash
$ git clone https://github.com/`YOUR_GITHUB_NAME`/PyTorch.git 
$ cd TensorFlow
```

### Step 3: Setup an upstream

변경 사항이 있을 경우, 쉽게 Pull할 수 있도록 이 Repo에 대한 링크를 설정해야 합니다.

```bash
$ git remote add upstream https://github.com/deeplearningzerotoall/PyTorch.git
```

저장소에 업데이트가 있는 경우 로컬 복사본과 repository 를 업데이트할 수 있습니다.

```bash
$ git pull upstream master && git push origin master
```

### Step 4: Make a branch

Master branch는 Pull Request들을 계속 병합되고 수정되기 때문에 Master branch를 직접 수정하지는 않는 게 좋습니다. 

그리고 의미 있는 이름으로 Branch를 만드는 걸 잊지마세요!


Example: 
```bash
$ git checkout -b hotfix/lab10 -t origin/master
```

새로운 Branch를 만든 후에 자유롭게 코드를 수정하세요!

**주의: 여러분 제안한 Issue와 관련이 없는 다른 것들을 고치고 마세요!**

만약에 다른 문제가 있다면, 따로 이슈를 제안하시길 바랍니다. 

### Step 5: Commit

이메일/사용자 이름을 설정하세요.

```bash
$ git config --global user.name "Sung Kim"
$ git config --global user.email "sungkim@email.com"
```

그리고 필요한 파일을 추가 후, Commit 하세요.
```bash
$ git add my/changed/files
$ git commit
```

Notes
* 다른 사람들도 알아 볼 수 있게 명확한 Commit 메시지를 쓰세요!

* 예시:
```text
Short (50 chars or less) summary of changes

More detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.

Further paragraphs come after blank lines.

  - Bullet points are okay, too

  - Typically a hyphen or asterisk is used for the bullet, preceded by a
    single space, with blank lines in between, but conventions vary here
```

### Step 6: (Optional) Rebase your branch

수정이 평소보다 더 오래 걸려서, 여러분의 레포지토리는 뒤쳐진 옛날 버전일 가능성이 높습니다. 항상 레포지토리를 최신 버전으로 동기화하세요.
```bash
$ git fetch upstream
$ git rebase upstream/master
```

### Step 7: Push

여러분의 repo를 push하기전에 ‘Autopep8’을 실행해주세요!

E501(최대 문자 줄 제한)을 제외한 모든 PEP8형식을 따라주세요. 

**잊지마세요, 가독성이 최우선입니다!**

* 예시:

```bash
$ autopep8 . -r -i --ignore E501
$ git push -u origin hotfix/lab10
```


### Step 8: Creating the PR
이제 여러분의 브라우저와 repo를 열면 "compare & pull request."라는 초록색 버튼을 보실 수 있습니다.

* 좋은 제목을 작성하세요.
* 여러분이 수정한 파일 이름만 쓰지마시고 자세하게 설명해주세요.
* **여러분이 했던 것과 여러분이 왜 했었는지를 설명해주세요.**
* 관련된 issue번호도 추가해주세요.

축하합니다! 여러분의 PR은 Collaborator들에게 검토받을겁니다.  
여러분의 PR이 CI Test도 통과했는지 체크하세요.

