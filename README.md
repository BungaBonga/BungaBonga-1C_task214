# BungaBonga-1C_task214
Алгоритм основан на известном алгоритме детектирования углов SUSAN. Сначала с помощью этого алгоритма находятся все углы около одной вершины или пересечения ребер. Далее все такие углы объединяются в один объект, находятся контуры изображений и по количеству контуров, так как нам на входе известно число вершин, мы находим число пересечений ребер.
При больших входных данных алгоритм может работать достаточно продолжительно.
