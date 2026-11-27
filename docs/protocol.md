# 실험 프로토콜과 사전 rubric

## 관찰 단위와 경계

한 쌍은 게임 하나의 체크포인트 하나에서 재개한 A와 B다. 체크포인트는 문제를 드러낸 observation 직후이며, 다음 thought를 생성하기 전이다. 게임 ID, 데이터 파일 SHA-256, 환경 seed, 초기 observation과 전체 이전 행동을 보관한다. 평가용 hidden facts와 admissible commands는 감사·채점에만 쓰며 모델 프롬프트에 넣지 않는다.

오류 후보는 실제 모델 행동에서 수집한다. 환경이 `Nothing happens.` 등의 실패를 명시했거나, 최근 최대 3행동에서 같은 명령이 두 번 이상 나오고 해당 구간의 symbolic state와 observation이 같아야 한다. 현재 반복 판정은 새 단서를 놓치지 않도록 구간 내 observation이 하나라도 다르면 제외하는 보수적인 규칙이다. 실패 행동을 인위적으로 삽입하지 않는다.

정상 후보 게임은 실행 전에 지정한다. 그 게임에서 실패 없는 첫 상태 변화가 일어난 직후를 정상 체크포인트로 삼는다. 먼저 실패가 나오면 그 게임을 정상 사례로 사용하지 않는다. 정상·오류 게임은 겹치지 않는다. 수집한 전체 게임 순서와 탈락 여부는 결과와 함께 남긴다.

## A/B 프롬프트

A는 체크포인트까지의 모든 thought·action·observation을 포함한다. B는 그중 thought **본문만** `[MASKED]`로 치환한다. thought 표식과 위치, 목표, 예시, action, observation과 나머지 프롬프트는 동일하다. 문자열 span 기반 테스트는 A의 thought 본문만 치환한 전체 문자열과 B가 정확히 같은지 검사한다.

재개 이후에는 양쪽 모두 새 짧은 계획 문장을 생성하고 각자의 이력에 그대로 추가한다. 별도 오류 점검 지시는 없다. 모델은 공개 계획 한 문장(지시상 최대 20단어)과 명령 한 개를 출력한다. 원 응답과 계획 단어 수를 기록하므로 지시 위반을 확인할 수 있다.

게임 쌍별로 A/B 순서를 seed 기반 무작위화해 `allocation.json`에 저장한다. 조건별 환경 인스턴스는 독립적이며 매번 초기 상태부터 저장된 행동을 재생한다. A/B의 같은 재개 행동 번호에는 같은 디코딩 seed를 적용한다. 전체 프롬프트를 매번 서버에 보내며, 게임·조건별 첫 요청 전에 `/slots/0?action=erase`를 호출하고 첫 요청의 `cache_n=0`을 검사한다. 같은 세션의 후속 요청에서만 `cache_prompt=true`로 KV cache를 재사용한다. 명령 문자열을 자동 교정하거나 실패 시 정답 행동으로 대체하지 않는다.

## rubric 작성 순서

1. `candidates.jsonl`의 마지막 실패 명령, observation, 관련 상태를 읽는다. A/B 실행이나 결과를 먼저 보지 않는다.
2. 실패 원인과 막힌 하위 목표를 구체적으로 쓴다. 상태 변화 없는 반복은 실제로 어느 하위 목표가 막혔는지 기록한다.
3. 원인을 해소하는 실제 행동 방향만 전략 수정으로 인정하도록 규칙을 작성한다. 단순한 행동 변경, 무관한 탐색, "고치겠다"는 선언은 제외한다.
4. 별도 환경을 새로 초기화하고 체크포인트까지 재생한 뒤, 복구 행동을 실행해 복구 상태가 도달 가능한지 확인한다. 이 확인은 적격성 검사이며 모델 성과에 포함하지 않는다.
5. 위 증거와 규칙을 `rubrics.json`에 저장한 후 `freeze`한다. 고정 뒤에는 사례나 기준을 A/B 성과에 맞춰 바꾸지 않는다.

rubric의 키는 `checkpoint_id`다. 다음은 **파일 형식 예시이며 실제 실험 결과가 아니다**. 행동·대상·상태 이름은 실제 체크포인트 증거로 바꿔야 한다.

```json
{
  "CHECKPOINT_ID": {
    "evidence": "take apple 1 from countertop 1 failed with Nothing happens; robot was at a different receptacle.",
    "blocked_subgoal": "pick up apple 1 from countertop 1",
    "detection_targets": ["apple", "countertop", "take apple 1"],
    "acceptable_directions": ["Navigate to countertop 1 before taking the apple."],
    "strategy_rules": [
      {"action_regex": "go to countertop 1", "require_state_change": true}
    ],
    "recovery_rules": [
      {"action_regex": "take apple 1 from countertop 1", "require_state_change": true}
    ],
    "recoverability_verified": true,
    "recoverability_evidence_file": "recovery_probes/CHECKPOINT_ID.json",
    "rationale": "Successful pickup repairs the failed pickup subgoal."
  }
}
```

`strategy_rules`와 `recovery_rules`는 OR 조건의 목록이다. 각 규칙의 `action_regex`는 실제 실행 행동에 적용하고, `require_state_change`가 참이면 행동 직전·직후 상태 해시가 달라야 한다. 실패 observation을 낸 행동은 통과하지 않는다. 상태로 더 정확히 판정할 수 있으면 `required_facts`, `forbidden_facts`, `observation_regex`를 추가한다. 현재 scorer가 허용하는 정확한 의미는 `src/react_ablation/scoring.py`의 규칙 함수를 따른다. 모호한 정규식으로 모든 이동이나 모든 행동을 허용하지 않는다.

정상 사례도 같은 JSON의 해당 ID에 정상 근거를 적는다. 오류 rubric의 여섯 필수 항목과 `recoverability_verified`는 `freeze`가 검사한다. `recoverability_verified: true`라는 값 자체는 검증 증거가 아니므로 실제 재생 로그를 함께 보관한다. 복구 상태는 원래 실패한 하위 목표의 해소를 뜻하며 전체 과제 성공과 구별한다.

## 별도 환경에서 복구 가능성 확인

아래 형태로 실제 후보를 재생해 사전에 정한 복구 행동을 실행할 수 있다. `RECOVERY_ACTIONS`에는 해당 오류 원인을 해소하는 행동만 넣는다. 생성한 로그를 검토한 뒤 rubric을 고정한다. 이 코드는 모델을 호출하지 않는다.

```python
import json
from pathlib import Path
from react_ablation.experiment import replay_checkpoint
from react_ablation.storage import read_jsonl, write_json

run_dir = Path("results/YOUR_RUN_ID")
checkpoint_id = "YOUR_CHECKPOINT_ID"
cp = next(c for c in read_jsonl(run_dir / "candidates.jsonl")
          if c["checkpoint_id"] == checkpoint_id)
RECOVERY_ACTIONS = ["replace with verified recovery action"]
env = replay_checkpoint(cp)
try:
    steps = []
    for action in RECOVERY_ACTIONS:
        steps.append({"action": action, **env.step(action)})
    write_json(run_dir / "recovery_probes" / f"{checkpoint_id}.json",
               {"checkpoint_id": checkpoint_id,
                "purpose": "pre-A/B recoverability verification only",
                "steps": steps})
finally:
    env.close()
```

## 채점과 눈가림

오류 감지는 첫 3행동 전에 나온 새 계획/응답만 본다. 실패 명령 또는 막힌 하위 목표를 구체적으로 지목해야 1이다. 감지 없이 행동을 제대로 바꿨으면 감지 0, 전략 수정 1이 가능하다. 전략 수정은 첫 5행동, 복구는 첫 10행동에서 사전 규칙과 실제 환경 증거로 판정한다. 전체 과제 성공은 최대 20행동에서 공식 환경 `won`을 사용한다.

정상 사례의 오탐은 체크포인트 전 이력에 오류가 있었다고 잘못 판단한 경우다. 재개 후 새로 실패한 행동을 정확히 지적한 경우는 오탐이 아니다. 어휘 기반 자동 언어 점수는 잠정값이며 조건명을 가린 검토 CSV에 점수와 근거를 남긴다. 검토자가 읽는 CSV에는 원래 A/B thought 이력을 넣지 않으며 매핑은 별도 파일에 둔다. 과거 thought가 마스킹됐다는 직접 표식이 생성 응답에 나오면 검토용 transcript에서 중립 표식으로 바꾸지만, 간접적인 조건 추정 가능성까지 완전히 제거되지는 않는다.

검토 후 `reviewer_detection_3` 또는 `reviewer_false_positive_3`에 0/1과 `reviewer_reason`을 쓰고 같은 `score` 명령을 실행한다. 이는 언어 판정을 변경하며 행동·상태 지표를 덮어쓰지 않는다. 보고서는 잠정 점수인지 검토 점수인지 구분한다.

## 통계와 해석 범위

각 지표에 A/B 분자와 분모, A−B 퍼센트포인트, A만 성공/B만 성공 건수를 보고한다. 오류 감지·전략 수정·복구의 분모는 배정한 모든 오류 체크포인트다. 완료되지 않은 실행을 분모에서 삭제하지 않는다. 과제 성공은 전체 배정 체크포인트, 오탐은 정상 체크포인트가 분모다.

부트스트랩은 게임 ID를 복원 추출하고 그 게임의 A/B를 함께 가져와 차이의 2.5·97.5 백분위수를 계산한다. 기본 10,000회, seed 20261127이다. 현재 설계는 게임당 한 쌍이므로 게임 단위와 쌍 단위가 일치한다. 작은 표본, 동일 점수와 0점이 많으면 구간이 넓거나 퇴화할 수 있다. 이것은 일반적 우열이나 무효과의 증거로 충분하지 않다.

이 실험은 고정된 Qwen2.5-1.5B-Instruct Q4_K_M, 선택된 ALFWorld 게임, 프롬프트와 디코딩 설정에 대한 탐색적 증거다. 과거 thought 내용 제거와 함께 토큰 수·토큰 위치도 바뀐다. KV cache 재사용으로 prompt 평가와 생성의 batch 크기가 달라질 때 logits가 비트 단위로 같다고 보장되지 않는다. 두 조건에 동일한 캐시 정책을 적용하고 조건 사이에는 지우지만, 하드웨어·빌드 간 수치 재현성은 별개의 제한이다. 정상·오류 선택 규칙, 짧은 이력, 규칙 기반 복구 정의와 잠정 언어 채점의 한계를 함께 해석한다. 모의 테스트 fixture와 사전 복구 probe를 실제 모델 벤치마크 결과로 합치지 않는다.
