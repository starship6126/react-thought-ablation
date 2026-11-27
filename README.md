# ReAct thought 이력과 오류 복구

ALFWorld의 실제 텍스트 게임에서 과거 `thought` 본문을 유지한 조건 A와 `[MASKED]`로 바꾼 조건 B를 비교하는 Python 실험이다. 모델의 `thought`는 매 행동 전에 출력하는 짧은 공개 계획 문장이다. 외부 모델 API와 API 키를 사용하지 않는다.

실험 모델은 **Qwen2.5-1.5B-Instruct GGUF Q4_K_M**으로 고정한다. `llama.cpp`의 `llama-server`를 CPU에서 실행하고 `127.0.0.1:18080`에만 바인딩한다. 공식 ALFWorld 평가 필터를 적용한 게임 수는 seen 140개, unseen 134개다. 모델에는 환경 observation만 제공하며 symbolic facts, admissible commands와 정답 행동은 제공하지 않는다.

## 설치와 실제 실행 확인

Linux x86-64, Bash, Git, curl, C/C++ 컴파일러와 Python 3가 필요하다. 설치 스크립트는 Python 3.11 가상환경, 고정된 Python 패키지, 공식 ALFWorld 텍스트 데이터와 공개 가중치를 내려받는다. CMake가 없으면 별도 빌드 가상환경에 설치한다.

```bash
git clone https://github.com/starship6126/react-thought-ablation.git
cd react-thought-ablation
bash scripts/setup_alfworld.sh
bash scripts/setup_runtime.sh
bash scripts/start_server.sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/smoke.py
```

`smoke.py`는 실제 seen 게임에서 모델의 계획과 행동을 생성하고 환경에 실행한다. 환경 재생, thought 마스킹, 파싱, 응답 시간, 문맥 길이를 검사해 `artifacts/setup/smoke.json`에 기록한다. 이 결과는 파일럿 성과에 포함하지 않는다. CPU·메모리·디스크는 `resources.json`, 모델과 실행기는 `runtime_manifest.json`, 환경과 데이터 해시는 `environment_manifest.json`에서 확인할 수 있다.

| 고정 항목 | 값 |
| --- | --- |
| Hugging Face 리비전 | `91cad51170dc346986eccefdc2dd33a9da36ead9` |
| 가중치 SHA-256 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| llama.cpp 커밋 | `5d7bb89f4719243dc52fe387bcb67c3518256565` |
| 서버 | CPU 4 threads, GPU layers 0, context 16384, parallel 1 |
| 디코딩 | temperature 0.2, top-p 0.9, top-k 40, min-p 0, repeat penalty 1 |
| 응답 상한 | 128 tokens; `Thought`와 `Action` 두 줄 형식 |
| 문맥 처리 | 자동 잘라내기·context shift 없음; 세션 안에서만 KV cache 재사용 |

## 파일럿: 오류 8개와 정상 4개, 12쌍

각 실행은 새로운 `run_id`를 사용한다. 아래 변수는 같은 셸에서 이후 명령에도 사용한다. 기존 실행 결과를 다른 설정으로 덮어쓰지 않는다.

```bash
RUN_ID="pilot_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/$RUN_ID"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli collect \
  --run-dir "$RUN_DIR" --split seen --errors 8 --normals 4 \
  --max-steps 25 --seed 20261127
```

수집은 고정 seed로 섞은 게임 순서에서 진행한다. 정상 후보 게임은 추론 전에 지정하고, 그 게임의 오류 없는 첫 상태 변화 직후를 선택한다. 오류 후보 게임에서는 명시적 명령 실패 또는 상태·단서 변화 없는 반복 행동의 첫 경계에서 멈춘다. 게임별 후보는 하나뿐이다. 후보의 마지막 observation 이후에는 아직 다음 thought를 생성하지 않는다.

**A/B 실행 전에** `candidates.jsonl`을 검토하고, 실패 근거·허용할 전략 수정 방향·복구 상태를 `rubrics.json`에 기록한다. 별도 환경에서 체크포인트를 재생해 복구 행동이 실제 실행되는지 확인한다. [프로토콜과 rubric 형식](docs/protocol.md)에 작성 방법이 있다. 후보가 복구 불가능하면 A/B를 시작하지 말고 제외 사유와 추가 수집 규칙을 기록한다. 결과를 본 뒤 사례를 교체하지 않는다.

```bash
PYTHONPATH=src .venv/bin/python -m react_ablation.cli freeze \
  --run-dir "$RUN_DIR" --rubrics "$RUN_DIR/rubrics.json"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli run --run-dir "$RUN_DIR"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli score --run-dir "$RUN_DIR"
```

`freeze`는 목표 개수, 실제 환경 재생, rubric 필수 항목을 검증하고 체크포인트와 프로토콜 해시를 고정한다. `run`은 조건 순서를 무작위화하고, 각 조건을 새 환경에서 초기 행동을 재생한 뒤 최대 20행동 동안 실행한다. A/B는 같은 게임 seed, 모델, 프롬프트 형식과 디코딩 설정을 사용한다. 각 조건의 재개 이후 thought는 해당 조건 이력에 그대로 쌓인다. 매 게임·조건 시작 전에 서버 slot cache를 지우고 첫 요청의 재사용 토큰 수가 0인지 확인한다. 이후 같은 세션에서는 CPU 실행 시간을 줄이기 위해 KV cache를 재사용한다.

중단 후에는 같은 `RUN_DIR`와 같은 명령을 다시 사용한다. 수집 및 조건별 JSON journal을 매 행동 후 저장하고, 완료된 조건은 `runs.jsonl`에 즉시 추가한다. 재개 시 기록된 행동을 환경에서 재생해 observation과 상태를 확인한다. 한 쌍이 끝날 때마다 보고서를 갱신한다. 완료 쌍 수는 `run`의 `pair_complete` 로그와 `report.md`에서 확인한다. 파싱·문맥 초과·재생 불일치는 실행을 중단하므로 정상 종료 여부도 확인해야 한다.

## 채점과 검토

| 지표 | 창 | 판정 |
| --- | --- | --- |
| 오류 감지 | 재개 후 3행동 | 새 계획/응답이 실제 실패 명령 또는 막힌 하위 목표를 구체적으로 지적 |
| 전략 수정 | 5행동 | 사전 등록한 원인 해소 방향의 행동을 실제 실행; 감지 점수와 독립 |
| 복구 | 10행동 | 사전 등록한 복구 행동·환경 상태 조건 충족 |
| 과제 성공 | 20행동 | 공식 환경의 `won` 판정 |
| 정상 사례 오탐 | 3행동 | 오류 없는 체크포인트의 이전 행동/목표에 대해 잘못된 오류 주장 |

오류 지표 분모는 배정된 모든 오류 체크포인트다. 아직 실행하지 못한 조건도 분모에서 빠지지 않으며, 진행 중 보고서에는 미완료 상태가 표시된다. 전략 수정과 복구는 환경 증거와 고정 규칙으로 채점한다. 막연한 재시도 언급과 실행하지 않은 수정 선언은 점수가 아니다.

언어 판단은 우선 보수적인 어휘 규칙으로 **잠정 점수**를 낸다. `blind_review.csv`에는 조건명과 과거 thought 본문이 없다. 검토자는 `reviewer_detection_3` 또는 `reviewer_false_positive_3`에 `0`/`1`, `reviewer_reason`에 근거를 입력한 뒤 `score`를 다시 실행한다. `reviewer_id`는 선택 사항이다. 검토자에게 `review_mapping.json`, `runs.jsonl` 또는 조건별 session 파일을 함께 주면 눈가림이 깨진다. 정상 사례에서는 재개 후 새로 발생한 실제 실패를 지적한 말을 과거 오류의 오탐으로 세지 않는다.

`report.csv`와 `report.md`는 조건별 분자·분모, A−B 퍼센트포인트, A만 성공/B만 성공 건수, 게임을 단위로 한 쌍별 부트스트랩 95% 신뢰구간을 제공한다. 기본 부트스트랩은 10,000회이며 seed를 기록한다. 작은 파일럿의 차이는 탐색적으로 해석한다. 이 결과를 다른 모델·양자화·프롬프트·환경이나 일반적인 ReAct 성능으로 확대 해석하지 않는다. 마스킹으로 프롬프트 길이가 달라지는 효과도 포함되어 있다. KV 재사용 시 batch 크기에 따라 부동소수점 결과가 달라질 수 있으므로 다른 하드웨어·빌드에서 비트 단위 동일한 출력을 보장하지 않는다.

## 본 실험: unseen 134개 게임

새 실행 ID에서 전체 134개 unseen 게임을 수집한다. 정상 게임과 오류 게임은 겹치지 않으며, 오류 게임당 첫 적격 오류 후보만 사용한다. 수집 후 A/B 결과를 보기 전에 정해진 게임 순서에서 복구 가능한 오류 80개와 정상 20개를 확정한다. 적격 사례가 부족하면 부족 수와 제외 사유를 보고하고 목표를 충족한 것으로 기록하지 않는다.

```bash
RUN_ID="unseen_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/$RUN_ID"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli collect \
  --run-dir "$RUN_DIR" --split unseen --errors 80 --normals 20 \
  --max-steps 50 --seed 20261127 --all-games
# 후보 검토와 별도 복구 재생 후 "$RUN_DIR/rubrics.json" 작성
PYTHONPATH=src .venv/bin/python -m react_ablation.cli freeze \
  --run-dir "$RUN_DIR" --rubrics "$RUN_DIR/rubrics.json"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli run --run-dir "$RUN_DIR"
PYTHONPATH=src .venv/bin/python -m react_ablation.cli score --run-dir "$RUN_DIR"
```

`--all-games`는 목표 수 도달 후에도 게임을 끝까지 조사하므로 오류 후보가 80개보다 많을 수 있다. `rubrics.json`에는 사전에 정한 순서와 적격성 규칙으로 선택한 정확히 80+20개만 포함한다. 이 명령은 본 실험을 별도로 실행하는 방법이며, 파일럿 결과가 본 실험 실행을 의미하지 않는다.

## 결과와 저장

실행 결과는 `results/<run_id>/`에 보관한다. 실제 완료 여부는 각 실행의 `report.md`를 기준으로 확인한다.

| 파일 | 내용 |
| --- | --- |
| `checkpoints.jsonl` | 고정 체크포인트, 원 이력, 실패 근거, rubric, 재생·마스킹 증거 |
| `runs.jsonl` | 실제 A/B 행동, 환경 observation·상태, 원 응답, 파싱·시간·토큰 정보 |
| `blind_review.csv` | 조건명을 제거한 언어 검토와 잠정 점수·근거 |
| `report.csv` | 지표별 분자·분모와 쌍별 비교 통계 |
| `report.md` | 완료 상태, 설정, 결과 해석과 한계 |

추가로 `candidates.jsonl`, `collection_games.jsonl`, `rubrics.json`, `allocation.json`, `freeze_manifest.json`, 조건별 journal과 검토 매핑을 보관한다. `.runtime/`의 모델·데이터와 `.venv/`는 Git에서 제외한다.

지정 Drive 프로젝트의 **실행 결과/`<run_id>`**에 위 다섯 파일을 원래 확장자로 업로드하고, 연결된 Drive에서 폴더·이름·개수 5개를 다시 조회해 확인한다. 인증은 Google Drive 연결로 수행하며 모델 API 키와 무관하다. Drive 업로드 실패는 실험 실패와 구분하고 로컬 결과를 보존한다. 이 저장소의 Python runner 자체에는 Drive 인증 정보가 없다.

다운로드가 차단되면 설치 로그의 실패 명령·도메인을 확인하고 접근이 복구된 뒤 해당 설치 명령을 다시 실행한다. 필요한 공개 호스트는 `github.com`, `release-assets.githubusercontent.com`, `huggingface.co`, Hugging Face가 리디렉션하는 `cas-bridge.xethub.hf.co`·`transfer.xethub.hf.co`·`cdn-lfs.huggingface.co`, `pypi.org`, `files.pythonhosted.org`이다. 세부 실패 증거는 `artifacts/setup/*failures*`에 남는다. TLS 검증을 끄거나 토큰을 코드에 넣을 필요가 없다.

원본: [ALFWorld](https://github.com/alfworld/alfworld), [공개 Qwen 가중치](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp).
