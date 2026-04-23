"""검색 결합 패키지.

이 패키지는 어휘 기반 점수, 밀집 임베딩 점수, 리랭커 점수를 결합하는
융합 검색 엔진을 외부에 노출한다. 공개 인터페이스를 좁게 유지하면
상위 계층은 여러 엔진을 직접 조합하지 않고 하나의 안정적인 검색 인터페이스에
의존할 수 있다.
"""

from src.retrieval.fused_search import FusedSearchEngine

__all__ = ["FusedSearchEngine"]
