# Stanford CS329A "Self-Improving AI Agents" 자율학습 커리큘럼

> 원본 강의: Stanford CS329A (Autumn 2025) — 자기 개선(self-improving) AI 에이전트
> 담당 교수: Aakanksha Chowdhery, Azalia Mirhoseini
> 본 문서는 개인 자율학습을 위해 원본 강의 스케줄을 바탕으로 재구성한 한국어 상세 커리큘럼입니다.
> 강의 원본: https://cs329a.stanford.edu/

---

## 목차

1. [강의 개요 및 학습 목표](#1-강의-개요-및-학습-목표)
2. [권장 사전 지식(Prerequisites)](#2-권장-사전-지식prerequisites)
3. [전체 일정 한눈에 보기](#3-전체-일정-한눈에-보기)
4. [주차별 상세 커리큘럼](#4-주차별-상세-커리큘럼)
   - [Part 1: LLM 자기 개선 기초 (1~4주)](#part-1-llm-자기-개선-기초-14주)
   - [Part 2: 고급 추론 및 계획 (5~8주)](#part-2-고급-추론-및-계획-58주)
   - [Part 3: 에이전트 확장 (9~12주)](#part-3-에이전트-확장-912주)
   - [Part 4: 애플리케이션 및 평가 (13~20주)](#part-4-애플리케이션-및-평가-1320주)
5. [핵심 논문 완독 가이드](#5-핵심-논문-완독-가이드)
6. [자율학습 실습 프로젝트](#6-자율학습-실습-프로젝트)
7. [평가 방식 및 자체 평가 기준](#7-평가-방식-및-자체-평가-기준)
8. [학습 리소스 및 도구](#8-학습-리소스-및-도구)
9. [학습 팁 및 커뮤니티](#9-학습-팁-및-커뮤니티)

---

## 1. 강의 개요 및 학습 목표

### 1.1 이 강의가 다루는 주제

이 강의는 **상호작용을 통해 스스로 계속 개선하는 AI 에이전트**의 최신 기법과 응용을 다룹니다. 크게 4단계로 구성됩니다.

| 단계 | 주제 | 핵심 질문 |
|------|------|-----------|
| 1 | LLM 자기 개선 기법 | 모델이 자신의 출력을 검증하고 개선하려면? |
| 2 | 도구·코드·메모리 확장 | LLM이 외부 도구를 쓰고 기억하도록 하려면? |
| 3 | 다단계 추론·계획 | 장기적 추론과 계획을 어떻게 학습·수행할까? |
| 4 | 평가 프레임워크 | 에이전트의 성능을 어떻게 견고하게 평가할까? |

### 1.2 학습 목표

이 커리큘럼을 완주하면 아래를 할 수 있어야 합니다.

- [ ] 최신 자기 개선 AI 에이전트 연구 논문을 스스로 읽고 요약할 수 있다
- [ ] test-time compute scaling, 검증자(verifier), RL을 통한 훈련시간 스케일링을 설명할 수 있다
- [ ] ReAct 같은 에이전트 프레임워크를 직접 구현할 수 있다
- [ ] 추론·계획·검색 기반 에이전트의 장단점을 비교 분석할 수 있다
- [ ] 에이전트 워크플로우에 대한 내 자신의 평가 벤치마크를 설계할 수 있다
- [ ] 주제 하나를 선정해 미니 연구 프로젝트(2~4주)를 기획·수행·발표할 수 있다

---

## 2. 권장 사전 지식(Prerequisites)

원본 강의는 대학원 세미나 수준이므로, 자율학습 전에 아래를 갖추는 것을 권장합니다.

### 2.1 필수 기본기

| 분야 | 필요한 수준 | 추천 공부 자료 |
|------|-------------|---------------|
| 머신러닝 기초 | 과적합, 손실함수, 옵티마이저, 평가 지표 이해 | Andrew Ng의 ML 코스, 《Deep Learning》(Goodfellow) 1~6장 |
| 딥러닝 | Transformer 구조, attention, self-supervised learning | **《The Annotated Transformer》**, 《Dive into Deep Learning》(d2l.ai) |
| LLM 프레임워크 | 따라하기: Hugging Face, transformers, vLLM | Hugging Face Learn 코스, Karpathy의 "Zero to Hero" 시리즈 |
| Python | PyTorch 기본, 간단한 맞춤 가중치 업데이트 | PyTorch 공식 튜토리얼 |
| 강화학습(RL) 기초 | MDP, policy gradient, PPO 개념 | Hugging Face Deep RL 강좌, Spinning Up in Deep RL(OpenAI) |
| 수학 | 확률, 정보이론 엔트로피, 선형대수 | 3Blue1Brown, CS229 부록 |

> **자세히 알아보기 — 강화학습(RL) 기초**:
> 자기 개선 에이전트의 핵심은 **정책(policy) 개선**입니다. RL에서 에이전트는 환경과 상호작용하며 보상을 최대화하도록 학습합니다. 이 강의에서 자주 등장하는 개념은 아래와 같습니다.
> - **정책(policy)**: 상태 → 행동 매핑 (LLM의 경우 프롬프트 → 텍스트)
> - **보상 모델(reward model)**: 강화학습 신호 제공
> - **PPO(Proximal Policy Optimization)**: LLM 정렬에 가장 널리 쓰이는 알고리즘
> - **GRPO(Group Relative Policy Optimization)**: DeepSeek가 도입한, 코드·추론에 유리한 효율적 대안

### 2.2 반드시 익힐 도구

- `Python 3.11+`, `PyTorch`, `transformers`, `vLLM/SGLang`
- `LangGraph` 또는 `AG2(구 AutoGen)` — 에이전트 오케스트레이션
- Git/GitHub — 논문 재현 코드 및 자신의 프로젝트 관리
- (선택) `Weights & Biases` — 실험 로깅
- (선택) GPU가 없다면 **Google Colab Pro / Lambda / RunPod** 활용

---

## 3. 전체 일정 한눈에 보기

> 원본 강의는 2025년 가을, 주 2회(월/금) 진행됩니다. 자율학습은 한 주차를 1주일 단위로 소화하는 것을 권장합니다.

| 주차 | 강의 주제 | 유형 | 핵심 키워드 |
|------|----------|------|------------|
| 1 | 강의 개요 | 기본 | 에이전트 정의, 자기 개선이란 |
| 2 | Test-time Compute Scaling | 기본 | 반복 샘플링, 추론시간 컴퓨트 |
| 3 | Robust Verification | 기본 | 검증자(verifier), step-by-step 검증 |
| 4 | 학습 신호로부터의 학습(도구·코드 사용) | 기본 | ReAct, RLEF, Constitutional AI |
| 5 | 다단계 추론/계획 | 심화 | LATs, SWiRL, ADaPT |
| 6 | 훈련시간 스케일링/RL | 심화 | STaR, DeepSeekMath, DAPO |
| 7 | 자기 개선 에이전트의 개방형 진화 | 심화 | AI Scientist, AlphaEvolve |
| 8 | 검색·딥리서치 에이전트 | 심화 | AlphaCode, Search-o1 |
| 9~12 | 게스트 강연 + 중간발표 | 심화 | 포스트트레이닝, 추론 모델 |
| 13 | 소프트웨어 엔지니어링 에이전트 | 응용 | CodeMonkeys, KernelBench |
| 14 | 메모리 확장 | 응용 | MemGPT, CacheBlend |
| 15~16 | 게스트 강연 (추론) | 심화 | AlphaProof, IMO 금메달 |
| 17 | 에이전트 평가 | 평가 | Long Tasks, GDPVal |
| 18~19 | 게스트 강연 (자율성, 로봇) | 응용 | Robotics, Real-world |
| 20 | 향후 연구 방향 | 마무리 | 오픈 리서치 질문 |

> 9~12, 15~19주차의 "게스트 강연" 부분은 자율학습 시 관련 논문과 블로그 글로 대체합니다. 각 주차에서 대체 자료를 안내합니다.

---

## 4. 주차별 상세 커리큘럼

> 각 주차는 **4단계 학습 플로우**를 따릅니다.
> **① 개념 익히기 → ② 논문 읽기(의 질문) → ③ 직접 구현/실험 → ④ 정리(요약글 쓰기)**
> 각 논문마다 "읽으면서 해야 할 질문"이 제시되어 있습니다.

---

## Part 1: LLM 자기 개선 기초 (1~4주)

### 1주차 — 강의 소개: 자기 개선 에이전트란 무엇인가

**학습 플로우**

1. 강의 개요 읽기: https://cs329a.stanford.edu/ , pastprojects 페이지 실제 프로젝트 참고
2. 자기 개선 정의하기: "모델이 자신의 결과를 검증하고, 보완하고, 재시도하는 능력"
3. 20주 커리큘럼 전체를 읽고 Part별 매핑 만들어보기

**기본 개념 정리**

- **AI 에이전트(agent)**: 목표를 달성하기 위해 환경과 상호작용하며 도구 사용·계획·기억을 수행하는 지능 시스템. 기존 LLM(일회성 텍스트 생성)과 달리 **여러 번의 추론과 행동(action)** 을 포함합니다.

- **자기 개선(self-improvement)** 의 세 가지 경로:
  1. **추론시간 개선(test-time)**: 답을 낸 뒤 검증하고 다시 시도 (monkeys/Archon)
  2. **훈련시간 개선(train-time)**: RL로 학습 자체를 개선 (STaR, DAPO)
  3. **구조적 개선**: 에이전트의 설계·구성 자체를 발견 (AI Scientist, AlphaEvolve)

- 핵심 트레이드오프: **성능 ↔ 추론 비용**, **성능 ↔ 평가 어려움**

**학습 산출물**: "자기 개선 에이전트란 무엇인가" 1페이지 요약문

---

### 2주차 — Test-time Compute Scaling: 추론시간 컴퓨트 확장

**개념 정리**

- 기존에는 모델 파라미터를 키우면 성능이 올라갔지만, 이제는 **추론(inference) 시점에 더 많은 컴퓨트(연산)를 쓰는 것**이 핵심 트렌드입니다.
- **Test-time compute 종류**
  1. **반복 샘플링(repeated sampling)**: 같은 문제에 여러 답을 생성 → 투표(voting)나 검증자로 최고 답 선택. 단순하지만 강력합니다.
  2. **베스트-오브-N(best-of-N)**: N개의 후보를 만들고 검증자(verifier)가 가장 좋은 것을 선정.
  3. **트리 검색(tree search)**: 답을 조각내 검색 트리를 구성 (MCTS 등).
  4. **Archon 프레임워크**: 검증자·집계·다중 라운드 등 추론시간 기법을 자동 설계/탐색하는 시스템.
  5. **Scaling law**: test-time compute와 오류율(error rate) 사이의 멱법칙(power law) 관계.

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 성능-컴퓨트 관계 | [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) | 왜 LLM의 오류율이 높아질수록 반복 샘플링의 이득이 커지는가? pass@k와 1−pass@k의 관계는? |
| 기법 자동 탐색 | [Archon: An Architecture Search Framework for Inference-Time Techniques](https://www.arxiv.org/abs/2409.15254) | 검증자, 집계, 재시도, 다중 라운드를 어떻게 조합해 최적 스택을 찾나? |
| 최적 컴퓨팅 배분 | [Scaling LLM Test-Time Compute Optimally...](https://arxiv.org/abs/2408.03314) | 컴퓨트를 "모델 파라미터" 대신 "추론시간 검색"에 쓸 때 언제 더 효율적인가? |
| 샘플링 스케일링 법칙 | [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) | 반복 샘플링의 스케일링은 (모델 간, 데이터셋 간) 어떤 법칙을 따르는가? |

**직접 실습해보기**

```python
# best-of-N 샘플링 데모
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "What is 17 * 23?"
msgs   = [{"role": "user", "content": prompt}]
ids    = tok.apply_chat_template(msgs, return_tensors="pt")

N = 8  # 샘플링 횟수
gen = model.generate(
    ids,
    do_sample=True,
    temperature=0.8,
    max_new_tokens=64,
    num_return_sequences=N,
)

answers = tok.batch_decode(gen, skip_special_tokens=True)
for i, a in enumerate(answers):
    print(f"[{i}] {a}\n")
# 공통된(일치하는) 답 = 다수결 → 실제로 "이 답이 옳다고 추가 검증"된 것으로 간주
```

**실험 과제**

1. 같은 프롬프트에 `temperature`를 0.2, 0.6, 1.0으로 바꿔가며 best-of-8을 수행하고 정답률 변화를 기록한다.
2. 어려운 문제(수학 추론, 코딩)에서 샘플 N을 1→16까지 늘리며 정확도 변화 그래프를 그린다.
3. 같은 논문 질문: "반복 샘플링은 왜 'pass@1은 낮으나 pass@k가 높은' 모델에게 유리한가?"

---

### 3주차 — Robust Verification: 견고한 검증

**개념 정리**

- **검증의 문제(verification gap)**: 생성(generation)은 잘 하지만 검증(verification)을 못하는 경우가 많습니다. "답을 만들기보다 답을 확인하는 것이 더 쉽다"는 가설이 핵심입니다.
- **검증자(verifier)** 종류
  1. **답 수준(OUTCOME)**: 최종 답만 보고 맞다/틀리다 판정
  2. **단계 수준(PROCESS)**: 각 추론 단계별로 올바른지 판정 (process reward model)
  3. **약한 검증자(weak verifier)**: 작은 모델로도 큰 모델의 답을 검증할 수 있는지 여부 (generation-verification gap 관련)
- 검증 이득을 다루는 핵심 질문:
  - "검증 스케일링 곡선이 생성 스케일링 곡선보다 유리한가?"
  - "단계별 검증이 최종 답 검증보다 나은가?"

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 약한 검증자 | [Shrinking the Generation-Verification Gap with Weak Verifiers](https://arxiv.org/abs/2506.18203) | 작은 모델(약한 검증자)도 큰 모델의 답을 검증에 유용하게 쓸 수 있는가? "이득 = 생성성능 − 검증성능"이 어떻게 달라지는가? |
| 검증으로 훈련 | [Training Verifiers to Solve Math Word Problems (Cobbe et al. 2021)](https://arxiv.org/abs/2110.14168) | GSMA8K에서 검증기 학습: generator 정책과 verifier를 분리하는 initWith annotations 방식을 되짚는다. |
| 단계별 검증 | [Let's Verify Step by Step (Lightman et al. 2023)](https://arxiv.org/abs/2305.20050) | process reward model이 outcome을 합친 것보다 왜 일반화에 좋은가? |
| 레이블 없는 단계 검증 | [Math-Shepherd: Verify and Reinforce LLMs Step-by-step](https://arxiv.org/abs/2312.08935) | 단계별 보상(stepwise reward)을 인간 레이블 없이 자동으로 만들려면? |

**직접 실습해보기**

1. GSMA8K 초급 문제 20개로 **best-of-N vs. 검증자 기반 선택**을 비교
2. 실행 코드 참고: 간단한 verifier는 "추론 단계를 LLM이 점수화"하는 방식으로 구성
3. 강화학습 연결: 단계별 보상으로 PPO를 돌리는 파이프라인 구성을 아키텍처 다이어그램으로 그려보기

**정리 질문**

- "생성 지니언스(generation skill)와 검증 스킬(verification skill)을 분리하면 왜 좋은가?"
- "검증 실패(부진)를 검출하려면 어떤 평가 지표가 필요한가?"

---

### 4주차 — Learning from Feedback with Tools/Code: 도구·코드 피드백 학습

**개념 정리**

- 도구 사용(tool use)이 왜 자기 개선의 핵심인가: LLM이 **실행 결과(execution feedback)** 를 관찰하면 자신의 실수를 수정할 수 있음
- **ReAct**: Reasoning(추론) + Acting(행동)을 번갈아 수행. 랭체인 시절부터 표준이 된 패턴.
- **RLEF(RL with Execution Feedback)**: 코드를 실제 실행하고, 실행 성공/실패 여부를 보상으로 강화학습
- **Constitutional AI**: 사람 피드백을 줄여가며, 모델이 정해진 원칙(헌법, constitution)에 따라 스스로 판단·수정
- **도구 인터페이스**: function calling, code interpreter, 검색 API

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 기본 패턴 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | 생각→행동→관찰(생각 위 관찰) 순서가 왜 긴 호스트 파이프라인보다 좋은가? |
| 코드 실행 피드백 + RL | [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089) | "실행 성공" 보상이 코드 LLM 정확도를 어떻게 높이는가? |
| 인간 피드백 줄이기 | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 규칙을 모델이 자신에게 적용해 스스로 개선(RLHF → RLAIF) |

**직접 실습해보기 — ReAct 구현**

```python
import json, requests

MATH_TOOLS = {
    "calculate": lambda expr: eval(expr, {"__builtins__": {}}),
    "search": lambda q: requests.get(f"https://api.duckduckgo.com/?q={q}&format=json").json().get("Abstract", ""),
}

def react(prompt, max_steps=5):
    # 흉내 구현: 단계마다 Thought -> Action -> Observation 순 출력
    for step in range(max_steps):
        # (실제로는 LLM에게 Thought/Action을 생성시키고 파싱)
        ...
```

**실전 실습 추천**

1. **간단한 산술 도구**에서만 동작하는 ReAct 에이전트 구현 → "Thought/Action/Observation" JSON 파싱
2. 탐색(tool call)이 안 되는 문제와 되는 문제에서 성능 비교
3. 도구 실행 결과(에러 메시지)를 관찰해 **재시도**하는 루프 디버깅

---

## Part 2: 고급 추론 및 계획 (5~8주)

### 5주차 — Multi-step Reasoning/Planning: 다단계 추론·계획

**개념 정리**

- 에이전트에서는 **한 번의 추론**이 아니라 **여러 단계의 추론을 이어붙이고 분기·병합**해야 함
- **LATS(Language Agent Tree Search)**: 기본 ReAct + 트리 검색(MCTS 변형)을 결합. Reasoning-Acting-Planning을 하나로
- **ADaPT**: 문제를 필요한 만큼만 분해(as-needed decomposition)
- **SWiRL**: 합성 데이터 생성 + 다단계 RL로 추론·도구사용 학습
- **SPRINT**: 추론과 실행을 병렬화(interleaved planning + parallel execution)
- **Adaptive branching**: 트리 분기 너비(wide = 더 넓은 탐색 / deep = 더 깊은 탐색) 선택

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 합성데이터+다단계 RL | [SWiRL: Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736) | 합성 데이터의 질을 어떻게 보장하며, 다단계 RL이 단순 RL과 무엇이 다른가? |
| 트리 검색 통합 | [Language Agent Tree Search Unifies Reasoning Acting and Planning (Zhou et al. 2023)](https://arxiv.org/abs/2310.04406) | LATS가 ReAct와 트리 검색을 어떻게 하나의 루프로 동작하게 하는가? |
| 병렬 실행 | [SPRINT: Interleaved Planning and Parallelized Execution in Reasoning Models](https://arxiv.org/abs/2506.05745) | 계획 수립과 도구 실행을 병렬화해 지연시간을 줄이는 방법 |
| 필요시 분해 | [ADaPT: As-Needed Decomposition and Planning with Language Models](https://arxiv.org/abs/2311.05772) | "구체적으로 필요한 만큼만" 분해하는 방법론이 전체를 먼저 분해하는 것보다 왜 나은가? |
| 분기 트레이드오프 | [Wider or Deeper? Adaptive Branching Tree Search](https://arxiv.org/abs/2503.04412) | 탐색 단계가 깊어질수록 분기(폭)를 넓힐지 조일지를 어떻게 결정하는가? |

**직접 실습해보기**

1. 최소 LATS 데모: ReAct + value estimation(검증자) → 상위 K개 후보 유지
2. ADaPT 구현: 도구가 바로 풀 수 있는 문제는 분해하지 않고, 어려운 문제만 재귀 분해
3. 비교 표 작성: ReAct vs LATS vs ADaPT를 (정확도, 스텝 수, 토큰 비용) 기준으로 비교

---

### 6주차 — Train Time Scaling / RL: 훈련시간 스케일링

**개념 정리**

- **훈련시간(training-time) 스케일링**: 모델을 학습시킬 때 컴퓨트를 많이 쓰면 성능이 올라가는 방향
- **STaR**: 모델이 스스로 시도한 추론(self-generated rationales)을 정답이면 유지, 틀리면 재시도 → 자기 부트스트랩
- **DeepSeekMath / DeepSeek-R1 계보**: GRPO(Group Relative Policy Optimization), RL을 통한 추론 강화
- **DAPO**: 대규모 오픈소스 RL 시스템. 불안정성(entry length penalty, token-level loss 등) 해결
- 멘탈 모델: "생각(추론)을 더 잘하기 위해 '검증 단계'를 훈련한다"

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 자기 부트스트랩 | [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465) | 왜 "좋은 답"을 사람이 아니라 모델 스스로 만들게 하면 일반화에 좋은가? 부트스트랩 순서를 도식화하라. |
| 수학 추론 RL | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | GRPO가 PPO보다 왜 메모리·효율면에서 유리한가? (보상 모델 대신 그룹 상대 비교) |
| 대규모 RL 시스템 | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) | 훈련 불안정성을 해결한 네 가지 기법(예: clip-higher, dynamic sampling)을 각각 설명하라. |

**중요 배경 지식 — PPO vs GRPO**

| | PPO | GRPO |
|---|---|---|
| 기준선(baseline) | 별도 value/critic 네트워크가 상태 가치 추정 | 같은 프롬프트의 그룹 내 상대적 점수 평균 사용 |
| 메모리 | critic(보상 모델) 파라미터 추가 필요 | critic 없음 → 메모리 절약 |
| 사용처 | 전통적인 RLHF | DeepSeek-R1, 수학·코딩 추론 |

**직접 실습해보기**

1. **trl(SF-CLI)을 사용한 GRPO 학습 데모** (가능하면)
   - `pip install trl` 로 GRPOTrainer 실습
   - 데이터: GSM8K 일부, 보상 함수로 "정답 코드 통과" 설정
2. STaR 유사 구현: 문제 → 샘플 추론 → 정답과 비교 → 맞으면 파인튜닝 데이터로 재사용
3. 학습 곡선 분석: reward가 수렴하는가? 오버피팅 되는가?

---

### 7주차 — Open-Ended Evolution of Self-Improving Agents: 개방형 진화

**개념 정리**

- 인간의 설계를 벗어나 **에이전트 시스템 자체가 스스로 진화**(설계 탐색, 도구 발명, 아이디어 생성)
- **AlphaEvolve(AlphaDev 계열)**: 코딩 에이전트가 알고리즘을 직접 설계
- **AI Scientist**: 과학적 발견의 전 과정(아이디어 → 실험 → 논문 → 리뷰) 자동화
- **ADAS(Automated Design of Agentic Systems)**: 에이전트의 코드 자체를 메타 에이전트가 개선

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 에이전트 자동 설계 | [Automated design of agentic systems](https://arxiv.org/pdf/2505.22954) | 메타에이전트가 소규모 라이브러리에서 다양한 에이전트를 만들면, 검색 공간을 어떻게 제약하는가? |
| 과학 자동화 | [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) | 전개적 과학 발견 파이프라인(idea→experiment→write→review)에 어떤 실무 문제가 있는가? |
| 알고리즘 설계 | [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) | RL로 알고리즘 자체를 개선할 때, 검증 검사(validatation)를 어떻게 설계하나? |

**정리 질문**

- "자기 개선이 '성능 개선'인지 '일반화 개선'인지 구분이 필요한 이유는?"
- "개방형 진화에서 사고(repressure)와 재발견(rediscovery) 문제를 다루는 방법은?"

---

### 8주차 — Self-improvement with Search & Deep Research Agents: 검색·딥리서치

**개념 정리**

- 강력한 에이전트의 핵심 기능: **정보 검색(search/deep research)과 최적 답변 합성**
- **AlphaCode 계보**: 코드 대회 준위에서 검색 기반 코드 생성
- **Search-o1**: 추론 모델(o1 계열)에 외부 지식 검색을 장착

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 대회 코드 생성 | [Competition-Level Code Generation with AlphaCode](https://arxiv.org/pdf/2203.07814) | 다양성 샘플링 + 필터링(테스트 실행)이 왜 경쟁 프로그래밍에서 필수인가? |
| 후속 보고서 | [AlphaCode 2 Technical Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf) | 검색/필터링 파이프라인과 "트리 탐색"을 어떻게 규모화했는가? |
| 검색+추론 | [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/pdf/2501.05366) | 추론 도중에 검색을 끼워 넣었을 때 지식 부족(shortage)을 어떻게 보완하는가? |

**직접 실습해보기 — 딥리서치 에이전트 구축**

- 하나의 클로즈드 도메인 질문(예: "PyTorch의 DataLoader에서 shuffle과 seed의 상호작용")에 대해
  1. 웹 검색(SerpAPI 또는 Tavily) 2회
  2. 각 결과를 요약·합성해 답변 생성
  3. 이중 검증(인용 근거 포함) 프롬프트 설계
- 산출물: `search → read → cite → synthesize` 파이프라인의 최소 구현

---

## Part 3: 에이전트 확장 (9~12주)

> 원본에서는 게스트 강연(Google DeepMind 등)이 진행된 주차입니다. 자율학습에서는 **아카이브 블로그/논문**으로 대체합니다.

### 9주차 — Post-training: Chatbots에서 Agents로 (Melvin Johnson 강연 대체)

**대체 학습 자료**

- [InstructGPT / RLHF 기술 블로그](https://openai.com/research/instruction-following)
- [DeepSeek-R1 수학 추론 논문](https://arxiv.org/abs/2501.12948)
- [Google의 "Agents Google" 블로그](https://ai.google.dev/agents) — 에이전트의 인식·추론·행동·도구 4요소

**학습 목표**

- 포스트트레이닝(post-training) 파이프라인을 다음 단계로 재구성할 수 있다: `SFT → RLHF/DPO → RLVR(코드 실행 검증) → 에이전트 파인튜닝`
- "챗봇(=단일 턴, 지시 따르기)"과 "에이전트(=외부 환경과 상호작용)"의 차이를 설명

### 10~12주차 — 중간 발표(Midterm Presentations) 대체: 소규모 문헌 리뷰 프로젝트

**자율학습 대체 활동**

- 기간: 3주
- 목표: 자기가 고른 주제 1개를 정해 **미니 문헌 리뷰 + 재현 실험**을 한 뒤, 10분 발표 슬라이드 + 3페이지 분량 리포트 작성
- 추천 주제:
  1. "Test-time compute vs. train-time RL: 어떤 조건에서 어느 쪽이 유리한가?"
  2. "검증자(verifier) 성능이 생성자(generator)에게 피드백을 주는 경로 3가지"
  3. "메모리: MemGPT류 long-term memory의 대안 비교"

---

## Part 4: 애플리케이션 및 평가 (13~20주)

### 13주차 — Agentic Frameworks for Software Engineering: SWE 에이전트

**개념 정리**

- 소프트웨어 엔지니어링(코드베이스 수정, 버그 수정, 벤치마크) 영역의 에이전트 프레임워크
- **CodeMonkeys**: 테스트 실행 반복을 통해 test-time compute를 확장
- **KernelBench**: LLM이 GPU 커널을 직접 최적화했을 때의 효율/성능

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 테스트 기반 반복 | [CodeMonkeys: Scaling Test-Time Compute for Software Engineering](https://arxiv.org/abs/2501.14723) | 함정: "테스트만 돌리면 통과" 하면 안 됨. 커버리지·회귀 유지가 왜 필요한가? |
| 커널 최적화 | [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/pdf/2502.10517) | 성능 지표(이론적 FLOP 대비)를 어떻게 설정하는가? |
| LLM 옵티마이저 | [Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces](https://arxiv.org/abs/2410.15625) | 에이전트—시스템 인터페이스(추론→컴파일러)를 어떻게 설계하는가? |

**직접 실습해보기**

- 간단한 버그 수정 에이전트: 코드베이스 1개(예: 계산기 CLI), 실패 테스트 3개 → 재현 + 수정 + 검증 루프

### 14주차 — Augmenting Agents with Memory: 메모리 확장

**개념 정리**

- 에이전트는 대화·상태·연속 학습을 위해 **메모리**가 필요
- **MemGPT(Letta)**: LLM을 운영체제로 취급 — OS 페이지(paging)처럼 컨텍스트를 계층화, shift-in/out
- **Cartridges**: 경량·범용 long-context 표현을 self-study(자기 학습)로 생성
- **CacheBlend**: RAG 서빙 시 캐시를 재사용(fusion)해 속도 개선

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 메모리 OS 메타포 | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | 페이지 결함(paging) 개념이 LLM 컨텍스트 창 관리에 어떻게 적용되나? |
| 경량 컨텍스트 | [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266) | 긴 문서를 압축한 표현을 자기 학습으로 만들면 RAG vs. 우수한가? |
| 캐시 퓨전 | [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](http://arxiv.org/abs/2405.16444) | 캐시된 KV 심층 결합이 왜 지연시간을 줄이는데 도움인가? |

**직접 실습해보기**

- 최소 MemGPT 데모: 2계층 메모리(주 메모리=context, 외부=DB)를 구현하고, 질문-응답 시 필요한 정보만 외부에서 읽어오도록 설계

### 15~16주차 — LLM Reasoning & Superhuman Reasoning (게스트 강연 대체)

**대체 학습 자료**

- Denny Zhou 강연: ["LLM Reasoning" 관련 기조 강연 영상/칼럼](https://arxiv.org/abs/2503.13320) (The Reasoning Gap in Reasoning Models)
- AlphaProof / AlphaGeometry 2 IMO 금메달 관련 [DeepMind 블로그](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level)
- o1/o3 계열 추론 체인(chain-of-thought) 논문

**학습 목표**

- 자연어 추론(reasoning)의 한계와 "추론 모델"이 주는 차이(전략 변화)
- 수학 증명(formal math)에서 검증자-생성자 루프가 우수한 이유
- 실전: 수학 경시문제에서 step-by-step reasoning을 도출해 보기

### 17주차 — Agentic Evaluations & Long-Horizon Tasks: 에이전트 평가

**개념 정리**

- 에이전트 평가는 단일 답 비교보다 **장기 테스크(long-horizon task)** 를 어떻게 완료하는지 봅니다
- **Measuring AI Ability to Complete Long Tasks**: 작업 길이(스텝 수)에 따른 성공률 곡선
- **GDPVal**: GDP로 연결된 실제 경제 가치 업무로 AI 성능 평가
- **DeepScholar-Bench**: 생성형 연구 종합(research synthesis)의 라이브 벤치마크·자동 평가

**논문 읽기**

| 주제 | 논문 | 읽으면서 답할 질문 |
|------|------|-------------------|
| 장기 태스크 | [Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499) | "태스크 길이"를 어떤 기준으로 나누며, 실패 지점은 어디인가? |
| 경제 가치 | [GDPVal: Evaluating AI Model Performance on Real-World Economically Valuable Tasks](https://arxiv.org/abs/2510.04374) | 평가가 실제 경제적 가치와 어떻게 연결되어야 하는가? |
| 리서치 벤치마크 | [DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis](https://arxiv.org/abs/2508.20033) | "살아있는" 벤치마크(신규 논문 자동 추가)의 설계는? |

**직접 실습해보기**

- 나만의 **평가 스위트(이벤트 기반)** 디자인:
  1. 사람이 평가할 10가지 작업(예: RAG Q&A, 코드 리팩터링, 웹 검색 리포트)
  2. 각 작업당 "완료 기준" 정의
  3. 같은 LLM에 여러 장치(온도, 도구)를 바꿔 성공률 비교

### 18~19주차 — Autonomy & Robotics (게스트 강연 대체)

**대체 학습 자료**

- Misha Laskin(이전 Reflection AI) 인터뷰·칼럼: "Planning in Natural Language"
- Physical Intelligence의 π0(파이 제로) [블로그](https://www.physicalintelligence.company/blog/pi0)
- 로봇 파운데이션 모델(embodied AI) 관련 서베이

**학습 목표**

- 실세계(환경 제어, 안전, 시뮬레이션→실제 시환)에서의 에이전트 난제 이해
- 멀티모달(시각 + 언어 + 행동) 에이전트 아키텍처 개요

### 20주차 — Future Research Areas: 향후 연구 방향

**학습 활동**

1. 이 코스에서 다룬 30+ 논문 중 재미있는 3개를 골라 "왜 이것이 아직 완성되지 않았는가"를 2문단으로 정리
2. 개방형 연구 질문 목록 작성:
   - "검증자 없이도 자기 개선이 가능한가?"
   - "보상 설계를 자동화할 수 있는가?"
   - "다중 모달 에이전트에서 신뢰할 수 있는 검사는?" 등
3. 최종적으로 **자기만의 3개월 후속 연구 계획** 1장 작성

---

## 5. 핵심 논문 완독 가이드

### 5.1 우선순위 우열 — 어떤 논문부터 읽을까

완전 초심자라면 아래 순서로 읽기를 권장합니다 (숙련자용은 주차별로 게시).

1. **ReAct** (시작 패턴, 가장 넓게 알려짐)
2. **Let's Verify Step by Step** (검증 개념)
3. **STaR** (자기 학습 개념)
4. **Scaling LLM Test-Time Compute Optimally** (스케일링 트레이드오프)
5. **DeepSeekMath / GRPO** (RL 적용)
6. 이후 주차별 진도에 맞춰 각각 선별

### 5.2 논문 읽는 법 (10단계 약식)

1. 제목·저자·초록(abstract) 확인
2. "실제 문제"가 무엇인지 1문장으로 재정리
3. "해결 방법" 그림(figure) 하나에 집중
4. 데이터셋·지표는? (어디에서 평가했는가)
5. "주장"과 "실제 실험" 사이 전개를 체크
6. baseline(비교 대상)이 무엇인지 확인
7. hyperparameter 결정을 따라가 보기
8. 한계(limitations) 파트 읽어보기
9. 내가 그 방법을 어디에 쓸 수 있을지 1가지 시나리오
10. 5줄 요약 카드(내 언어로) 작성 → 카드 생성

> 배경 지식이 필요하면 번역된 요약(및 짱 '한국어 기술 블로그')을 먼저 읽고, 원문을 다시 읽기를 추천합니다.

### 5.3 추천 요약 카드 템플릿

```
# 카드 제목
- 논문 링크: ...
- 핵심 질문: ...
- 방법 한 줄: ...
- 중요한 실험(수치): ...
- 나의 인사이트: ...
```

---

## 6. 자율학습 실습 프로젝트

> Stanford 원본 과제(홈워크 1~3, 프로젝트)는 미공개입니다. 아래는 동일한 스킬을 기르기 위한 **대체 실습**입니다.

### 프로젝트 A (홈워크 1 대체): "모니키 아키텍처" 비교 그리드
- 기한: 1주
- 내용: best-of-N vs. 검증자 선택 vs. 단순 답변 — 50개 수학 문제에서 정확도·비용 비교표와 그래프
- 산출물: Python 스크립트 + 1페이지 보고서

### 프로젝트 B (홈워크 2 대체): "ReAct 에이전트 리틸"
- 기한: 1주
- 내용: 검색/계산 도구를 가진 ReAct 에이전트를 만들고, 계획 실패→재시도 로직 개선
- 산출물: 실행 가능한 코드 + 데모 로그

### 프로젝트 C (홈워크 3 대체): "메모리 에이전트"
- 기한: 1.5주
- 내용: 2계층 메모리(MemGPT식)를 구현하고, 장기 대화 시 질문 기억률 개선 실험
- 산출물: 3개 시나리오 실험 결과 + 사후 분석

### 프로젝트 D (파이널 프로젝트 대체): "자기 주제 리서치 + 재현"
- 기한: 4주
- 내용: 위 주제 중 하나를 심화하거나 자유 주제 1개 선정. 논문 1편을 엄밀히 재현(또는 변형)하고 3페이지 보고서 + 10분 발표
- 추천: "약한 검증자가 강한 생성자를 검증할 수 있을까?"를 GSM8K/Conda 셋으로 실험

### 프로젝트 E (포스터 대체): 
- 기한 1일: 프로젝트 D 결과를 1페이지 포스터(PNG 또는 Markdown)로 정리 — 마지막 주차에 "가상 포스터 발표"로 서로 리뷰

---

## 7. 평가 방식 및 자체 평가 기준

| 항목 | 원본 비중 | 자율학습 자체 점검 방법 |
|------|----------|------------------------|
| 홈워크 1 | 15% | 과제 A 완료 여부 + 질문 10개 통과 |
| 홈워크 2 | 15% | 과제 B, 코드 리뷰 |
| 홈워크 3 | 20% | 과제 C, 실험 재현 가능성 |
| 프로젝트 제안서 | 2.5% | 과제 D에서 아이디어 1paragraph |
| 중간발표+리포트 | 10% | 10분 발표 연습 + 3페이지 요약 |
| 파이널 프로젝트 | 35% | 과제 D 결과물 |
| 포스터 | 2.5% | 과제 E |

**자체 점검 자가 평가 루틴**

- 매주 토요일 30분: 이번 주 배운 것을 3문장으로 설명할 수 있는가?
- 매월 1회: 지난달 읽은 논문을 다시 보지 않고 5줄 요약?

> 원본 평가 기준도 참고: 벌칙 / 감점 정책(늦은 제출 하루 25% 감점), 4개의 무료 늦음 일수 등.

---

## 8. 학습 리소스 및 도구

### 8.1 강의 공식 자료
- 강의 홈: https://cs329a.stanford.edu/
- 지난 프로젝트: https://cs329a.stanford.edu/pastprojects.html
- 논문 목록: 본 문서 주차별 표에 표시

### 8.2 필수 도구 설치 리스트 (Windows 기준)

```powershell
# Python + 패키지 (권장: conda 또는 pyenv)
conda create -n cs329a python=3.11 -y
conda activate cs329a
pip install torch transformers accelerate datasets peft
pip install vllm sglang
pip install langgraph langchain-openai
pip install trl  # GRPO 실습용
pip install matplotlib jupyterlab
pip install pytest  # 코드 실험 자동화
```

### 8.3 대체 학습 자료 (게스트 강연 대체)

| 원본 게스트 | 대체 자료 |
|------------|----------|
| Melvin Johnson (포스트트레이닝) | InstructGPT 블로그, 디지테크의 RLHF 튜토리얼 |
| Denny Zhou (LLM 추론) | [The Reasoning Gap in Reasoning Models](https://arxiv.org/abs/2503.13320), o1 논문 |
| Thang Luong (AlphaProof) | IMO에서 금메달 수준 논문·블로그 |
| Misha Laskin (자율성) | "Planning in Natural Language" 인터뷰 |
| Danny Driess (로보틱스) | Physical Intelligence π0 블로그 |

### 8.4 참고 커뮤니티
- Hugging Face 메일링 리스트/포럼 (GRPO, R1 재현 프로젝트 활성)
- r/LocalLLaMA (커뮤니티 재현·래그 경험)
- arXiv `cs.AI`, `cs.CL` 주간 업데이트

---

## 9. 학습 팁 및 커뮤니티

### 9.1 스터디 루틴 (주간 템플릿)

| 요일 | 학습 활동 | 예상 시간 |
|------|----------|----------|
| 월 | 개념 정리 + 배경 지식 | 2시간 |
| 화 | 논문 1 (읽기 + 요약 카드) | 2시간 |
| 수 | 논문 2, 3 (가볍게 훑기 + 비교) | 2시간 |
| 목 | 직접 실습(코드) | 2~3시간 |
| 금 | 실험 분석 + 이슈 정리 | 1.5시간 |
| 주말 | 주차 요약문(3문장) + 복습 | 1시간 |

### 9.2 자주 하는 실수 & 팁

1. **도구(구현)에만 시간을 쓰지 말 것** — 개념→논문→실험의 균형
2. **논문을 깊이 파기 전에 배경(e.g., PPO, 검증기)을 먼저 확실히** — 중간에 막히면 뒤로 돌아가기
3. **"재현"은 로깅(메트릭)부터** — 실험을 무작정 돌리지 말고 `random seed`, `wandb` 사용
4. **수치(정확도) 비교를 의심할 것** — 같은 벤치마크라도 프롬프트 온도가 결과를 바꿈
5. **GPU가 없어도 작은 모델로 충분** - 1B~7B 모델로 모든 개념 실증 가능 (비용 절감)
6. **한 주에 논문 3~5편은 벅찰 수 있음** — 핵심 1편씩만 깊게, 나머지는 요약만

### 9.3 완료 후 다음 단계
- Kaggle / Hugging Face 벤치마크에 참여
- 최신 서베이 계속 추적: "Awesome AI Agents", "Awesome Self-Improvement" GitHub 레포
- 포트폴리오: 재현 프로젝트 1개를 README + Colab 노트북으로 공개

---

## 부록 A: 20주 전체 로드맵 (체크리스트)

- [ ] 주차 1: 에이전트/자기 개선 개념 정리
- [ ] 주차 2: 반복 샘플링(best-of-N) 실습
- [ ] 주차 3: 검증자(verifier) 실습 + generation-verification gap
- [ ] 주차 4: ReAct 에이전트 첫 구현
- [ ] 주차 5: LATS/ADaPT 검색 비교
- [ ] 주차 6: GRPO/RL 미니 실험
- [ ] 주차 7: 진화·개방형 설계(ADAS) 개념
- [ ] 주차 8: 딥리서치 에이전트 구현
- [ ] 주차 9~12: 미니 문헌 리뷰 + 발표
- [ ] 주차 13: 코드 에이전트 벤치마크 이해
- [ ] 주차 14: 메모리(MemGPT) 실습
- [ ] 주차 15~16: 추론 모델 깊이 이해
- [ ] 주차 17: 나만의 평가 스위트 설계
- [ ] 주차 18~19: 로봇/실세계 에이전트 개념
- [ ] 주차 20: 후속 연구 계획 작성

> **마지막 안내**: 원본 강의의 홈워크·강의 녹화는 공개가 아닐 수 있습니다. 동일한 커리큘럼을 자습할 때는 최신 벤치마크(paper)를 항상 함께 확인하세요. 원본에서 2025년 가을 기준이므로 향후 몇 년 안에 내용이 바뀔 수 있습니다.
>
> 화이팅 :) 20주가 차지 않게 매일 1시간씩 꾸준히 하면 충분히 소화 가능합니다.