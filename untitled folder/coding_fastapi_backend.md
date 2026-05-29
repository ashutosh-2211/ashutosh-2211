# FastAPI / Backend / Coding
# Easy → Hard | Ask in order, stop when candidate struggles

---

## FastAPI Basics

**Q1. [Easy]** What is the difference between a synchronous and asynchronous endpoint in FastAPI, and when does it matter?

> `def` endpoint blocks the thread while waiting (DB, HTTP calls). `async def` releases the thread during waits so other requests can run. Matters under load — sync endpoints bottleneck quickly. Use `async def` for any I/O-bound work.

---

**Q2. [Easy]** Why use `Depends()` for database sessions instead of creating a session inside the function?

> `Depends()` scopes the session to the request lifecycle — auto-closed and cleaned up after response. Creating inside the function risks leaks if an exception is thrown before close. Also makes testing easy — swap the dependency with a mock.

---

**Q3. [Easy-Medium]** You have a `/profile` endpoint that calls the DB 3 times to fetch user info, user settings, and user roles. How do you reduce DB load?

> Single query with JOINs. Or fetch all 3 in parallel using `asyncio.gather()` if they're independent. Cache the result in Redis with a short TTL if this endpoint is hit frequently.

---

**Q4. [Medium]** Your FastAPI app's connection pool is exhausting. DB connections are piling up and requests timeout. What's wrong and how do you fix it?

> Sessions not being closed — likely no try/finally in the DB dependency. Fix with a generator dependency that always closes the session. Also check pool size vs. expected concurrency. Set `pool_size` and `max_overflow` in SQLAlchemy config.

```python
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except:
            await session.rollback()
            raise
```

---

**Q5. [Medium]** APIs are slow. No profiler is set up. Users report it's worse around midnight. Where do you start?

> Midnight spike → likely a scheduled job hammering the DB or holding locks. Check slow query logs. Look at connection pool wait times. Add OpenTelemetry or `py-spy` on a live pod. Scheduled batch jobs and API traffic should never share the same DB resource pool.

---

## Idempotency & Distributed Consistency

**Q6. [Easy-Medium]** A user hits "Submit Order" twice because the first request was slow. Both requests reach your server. The order gets created twice. How do you prevent this?

> Idempotency key — client generates a unique ID per order attempt (e.g., UUID). Server checks if this key was already processed. If yes, return the stored result. If no, process and store result with the key. Add a unique constraint on the key column to catch races at the DB level.

---

**Q7. [Medium]** You have 1,000 replicas of your FastAPI app. A user hits "Transfer Money" and due to a network blip, the client retries. Both requests hit different replicas at the same time. How do you ensure the transfer happens exactly once?

> Idempotency key in the request. On each replica, before processing:
> 1. Try to insert the idempotency key into a `processed_requests` table with a unique constraint.
> 2. If insert succeeds → this replica owns the request, process it.
> 3. If insert fails (duplicate key) → another replica already processed it, return stored result.
> The DB unique constraint is the single point of truth — no inter-replica coordination needed.

---

**Q8. [Medium]** Two users simultaneously try to book the last seat on a flight. Both see "1 seat available." Both book. Now you've oversold. Fix this without using distributed locks.

> Optimistic locking — add a `version` column to the seat row. Read version with the seat. On update: `WHERE id=X AND version=Y`. Only one succeeds; the other gets 0 rows updated → retry or return "sold out." Alternatively use `SELECT FOR UPDATE` to lock the row for the transaction duration.

---

**Q9. [Medium-Hard]** 1,000 FastAPI pods all consume from the same Kafka topic — an "order placed" event should trigger exactly one inventory deduction. How do you ensure exactly-once processing across all pods?

> Kafka consumer groups handle partition assignment — each partition is consumed by exactly one pod at a time. Make consumers idempotent: before deducting inventory, check if the `event_id` is already in a `processed_events` table. If yes, skip. Use the transactional outbox pattern: write the inventory change and mark the event as processed in the same DB transaction. Never ack Kafka until both writes succeed.

---

**Q10. [Hard]** With 1,000 replicas and Redis as a distributed lock, a pod acquires a lock, does the work, but crashes before releasing. The lock has a 30-second TTL. For 30 seconds, no other pod can process that request. What's the better design?

> Optimistic concurrency in the DB (version fields + conditional updates) is more resilient than distributed locks — no lock to get stuck. If you must use Redis locks, use Redlock across 3+ Redis nodes for fault tolerance, set a short TTL, and make the underlying operation idempotent so if it runs twice due to a lock race, the second run is a no-op. The real answer: design operations to be naturally idempotent so locks become optional.

---

**Q11. [Hard]** A user clicks "Generate Report" 10 times in 5 seconds. The report takes 10 minutes to generate. You have 1,000 replicas. How do you ensure only one job runs and all 10 clicks get the correct status?

> On first request: atomically set a Redis key `report:{user_id}:{report_type}` = `{job_id, status: queued}` using `SET NX EX 600`. If key already exists, return the existing job_id. Enqueue the job only if you set the key (i.e., you won the race). All 1,000 replicas check the same Redis key. Expose `GET /report/status/{job_id}`. On job completion, update the Redis key and store the result.

---

## NLP / AI Coding Scenarios

**Q12. [Medium]** You need semantic search over 1 million company PDFs — HR, Legal, Finance. Users ask natural language questions, expect answers in under 500ms. Design it.

> Chunk PDFs (512 tokens, 50-token overlap). Embed with a sentence transformer. Store in vector DB (pgvector, Pinecone, Weaviate). At query time: embed question → ANN search → metadata filter by department → rerank top-20 → top-5 to LLM. Cache frequent queries in Redis.

> Follow-up — scales to 100M docs? Shard vector DB by department. HNSW index. Async ingestion queue.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
query_vec = model.encode("travel reimbursement for contractors")
# pass to vector DB for ANN search
```

---

**Q13. [Medium]** 500k support tickets arrive daily. Auto-detect duplicates before creating a new ticket. Implement core logic.

> Embed incoming ticket. Query vector DB for nearest neighbors. Cosine similarity > 0.85 → duplicate, return existing ticket. Below threshold → new ticket, index its embedding.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
e1 = model.encode("Unable to login after password reset")
e2 = model.encode("Forgot password and cannot access account")
score = cosine_similarity([e1], [e2])[0][0]  # ~0.92 → duplicate
```

> Follow-up — 10M tickets? ANN via vector DB. Never brute-force pairwise at scale.

---

## Coding Tasks

**Q14. [Easy — 2yr level]** Write a PyTorch CNN to classify MNIST digits. Use 3 conv layers followed by fully connected layers.

> Candidate should write a Sequential model with Conv2d → ReLU → MaxPool blocks, then flatten and linear layers. Key things to check: correct in/out channels, kernel sizes, flatten before linear, output size 10.

```python
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 64x7x7 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),           # 128*7*7 = 6272
            nn.Linear(6272, 256),
            nn.ReLU(),
            nn.Linear(256, 10)      # 10 digit classes
        )

    def forward(self, x):
        return self.model(x)
```

> What to check: Do they know input channels = 1 (grayscale)? Do they flatten before linear? Do they output 10 logits? Do they know MaxPool halves spatial dims? Bonus: BatchNorm, Dropout, softmax discussion.

---

**Q15. [Medium — 3-4yr level]** HR salary dataset, 100k rows. Get department averages, top 10 earners, 95th percentile salary, and flag salary outliers.

```python
import pandas as pd

# Department average
df.groupby("department")["salary"].mean()

# Top 10
df.nlargest(10, "salary")

# 95th percentile
df["salary"].quantile(0.95)

# Outliers via IQR
q1, q3 = df["salary"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df["salary"] < q1 - 1.5*iqr) | (df["salary"] > q3 + 1.5*iqr)]
```

> Follow-up — 500M rows? PySpark, partition by department, window functions for percentile.

---

**Q16. [Hard — 5yr level]** Secure a public FastAPI app handling enterprise docs and PII. Endpoints: `POST /chat`, `POST /upload`, `GET /history`. Multi-tenant — customers must never see each other's data.

> JWT auth via `Depends()`. OAuth2 for SSO. RBAC. Rate limiting (slowapi or API Gateway). WAF. Secrets in Vault. HTTPS only. Audit logging. For multi-tenancy: tenant_id in JWT claims, validated on every query, row-level security in Postgres. Never trust URL params for tenant context.

```python
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return {"user_id": payload["sub"], "tenant_id": payload["tenant_id"]}

@app.get("/history")
def history(user=Depends(get_current_user)):
    return get_history(user["user_id"], tenant_id=user["tenant_id"])
```

---
