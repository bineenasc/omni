# legacy/ — código não usado pela versão atual

Arquivos mantidos apenas como referência histórica. **Não são importados** pela
versão atual do treino.

## `lidar_cnn.py`

Extrator de features `LidarCNNExtractor` (CNN 1-D sobre o LiDAR + MLP) para uma
política `MultiInputPolicy` com espaço de observação `Dict` (`{"lidar", "vector"}`).

Foi abandonado porque:

1. A versão atual usa observação **plana** `Box(23,)` com `MlpPolicy` — não há
   mais espaço `Dict`, então este extrator é incompatível com o env atual.
2. Em testes, o modelo com CNN nunca marcou gol em 611k passos: o bônus de
   contato dominava o sinal e o robô aprendeu a *hover* perto da bola. As
   features analíticas (GPS + direções em frame local) treinam melhor.

Para reativar seria necessário voltar a observação para `Dict` e usar
`policy="MultiInputPolicy"` com `policy_kwargs=dict(features_extractor_class=LidarCNNExtractor)`.
