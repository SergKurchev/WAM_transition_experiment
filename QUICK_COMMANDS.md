# Быстрые команды для запуска

## 🚀 Полный цикл запуска (3 терминала)

### Терминал 1️⃣ — Запуск стека (на СЕРВЕРЕ)

```bash
# Подключитесь к серверу
ssh -p 2221 root@176.109.83.84

# Перейдите в папку
cd ~/skurchev/workspace/wam-stack

# Запустите стек с моделью unifolm
WAM_MODEL=unifolm bash scripts/deploy.sh
```

---

### Терминал 2️⃣ — Смотреть логи (на СЕРВЕРЕ)

```bash
# Подключитесь к серверу (новое окно)
ssh -p 2221 root@176.109.83.84

# Перейдите в папку
cd ~/skurchev/workspace/wam-stack

# Смотрите логи хлопания
docker logs -f wam-inference
```

---

### Терминал 3️⃣ — SSH туннель для видео (ЛОКАЛЬНО на вашем ПК)

```bash
# Создайте туннель (используйте SSH ключ!)
ssh -N -L 6081:localhost:6080 -p 2221 -i ~/.ssh/id_ed25519 root@176.109.83.84
```

**Затем откройте браузер:** http://localhost:6081

---

## 🛑 Остановка стека

На сервере (терминал 1):
```bash
docker compose down
```

---

## 🔍 Отладка

### Проверить статус сервисов
```bash
docker compose ps
```

### Смотреть логи Isaac Sim
```bash
docker logs -f wam-isaac-sim | tail -50
```

### Смотреть логи GEAR-SONIC
```bash
docker logs -f wam-gear-sonic | tail -50
```

### Смотреть логи ROS2 Bridge
```bash
docker logs -f wam-ros2-bridge | tail -50
```

---

## 📝 Примечание

- **SSH ключ:** `~/.ssh/id_ed25519` (используется для аутентификации)
- **Порт сервера:** 2221
- **IP сервера:** 176.109.83.84
- **Локальный браузер:** http://localhost:6081 (форвардится на удаленный 6080)
