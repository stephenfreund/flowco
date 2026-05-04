# Deployment

The app supports two deployment targets on AWS:

- **Lightsail Container Service** (current production) — single-node, ~$15/mo.
- **AWS Fargate / ECS** (legacy, kept for rollback) — auto-scaling, ~$50/mo.

Switching between the two is a DNS change plus a few service-state toggles. No code changes are required to switch directions.

---

## 1. Original config — Fargate

### Resources
- **ECS cluster**: `flowco-cluster` (region `us-east-1`)
- **ECS service**: `flowco-service`
- **Task definition**: `flowco-task` (family), 1 vCPU / 2 GB, awsvpc networking
- **ECR repository**: `925527669208.dkr.ecr.us-east-1.amazonaws.com/flowco-app`
- **ALB**: `flowco-alb-1197595225` (`dualstack.flowco-alb-1197595225.us-east-1.elb.amazonaws.com`)
- **ACM cert**: ALB cert for `go-flow.co` + `www.go-flow.co` (validated via Route 53 CNAMEs prefixed `_0145993f...` and `_73f39f6bf...`)
- **CloudWatch log group**: `/ecs/flowco-task`
- **IAM**: `ecsTaskExecutionRole`

### DNS
- Authority: **Route 53** hosted zone `go-flow.co`
- `go-flow.co` apex: A-record ALIAS → ALB
- `www.go-flow.co`: A-record ALIAS → ALB

### Repo files
- `.github/workflows/deploy.yml` — release-triggered workflow that builds the image, pushes to ECR, registers a new task definition, and updates the ECS service.
- `.github/ecs/flowco-task.json` — task definition template used by the workflow.

### Cost
Roughly **~$50/mo** for one task running 24/7 plus the ALB.

---

## 2. New config — Lightsail Container Service

### Resources
- **Container service**: `flowco` (region `us-east-1`, power **Small** = 0.5 vCPU / 2 GB, scale 1)
- **Service public hostname**: `flowco.gss8b1ryfh8jc.us-east-1.cs.amazonlightsail.com`
- **Image registry**: Lightsail's internal registry (`aws lightsail push-container-image` uploads here; image refs look like `:flowco.app.<n>`)
- **TLS cert**: Lightsail-managed cert for `go-flow.co` + `www.go-flow.co`, attached to the service
- **DNS zone**: Lightsail-hosted zone for `go-flow.co`

### DNS
- Authority: **Lightsail DNS zone** for `go-flow.co`. Nameservers are set on the registered domain in Route 53 → Registered domains.
- Apex `@`: A-record alias → container service `flowco` (Lightsail apex-alias feature)
- `www`: CNAME → service public hostname

### Repo files
- `.github/workflows/deploy-lightsail.yml` — release-triggered workflow that builds the image, pushes via `aws lightsail push-container-image`, and creates a new container service deployment.

### Pinned dependency
- `requirements.txt` pins `google-auth-oauthlib<1.2.0`. Newer versions enable PKCE by default in `Flow.authorization_url()`, but `src/flowco/ui/authenticate.py` constructs a fresh `Flow` on the OAuth callback that has no `code_verifier`, which breaks the token exchange. Either keep the pin or fix the auth flow to persist `flow.code_verifier` across the redirect via `st.session_state`.

### Cost
Roughly **~$15/mo** (Small container tier). No ALB, no separate static IP.

---

## 3. Migration: Fargate → Lightsail

This was performed on 2026-05-04. Steps below are the as-run procedure, useful as a reference if migrating again.

### One-time AWS setup
1. **IAM permissions** — attach to the GitHub Actions IAM user (whose access keys are in GitHub Secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": [
         "lightsail:GetContainerServices",
         "lightsail:GetContainerImages",
         "lightsail:GetContainerServiceDeployments",
         "lightsail:RegisterContainerImage",
         "lightsail:CreateContainerServiceDeployment"
       ],
       "Resource": "*"
     }]
   }
   ```
2. **Create the container service** — Lightsail console → Containers → Create container service. Name `flowco`, region `us-east-1`, power Small, scale 1.

### Deploy
3. Run the **Deploy to AWS Lightsail** workflow in GitHub Actions (`workflow_dispatch`) on `main`. First push to Lightsail's image registry takes a few minutes. The workflow creates the initial deployment with a public endpoint on port 80. Verify at the auto-assigned `https://flowco.<id>.us-east-1.cs.amazonlightsail.com/` URL.

### TLS cert
4. Lightsail console → `flowco` container service → Custom domains → **Create certificate** for `go-flow.co` + `www.go-flow.co`.
5. Validation: Lightsail returns two `_<hash>.acm-validations.aws` CNAMEs. Add them to Route 53 hosted zone `go-flow.co` (Name field takes only the prefix, not the zone suffix). Wait ~5 min for cert to flip to **Valid**.
6. Custom domains → **Attach domain** → tick both → Attach.

### DNS migration to Lightsail
7. Lightsail → Domains & DNS → **Create DNS zone** → `go-flow.co`. Note the 4 Lightsail nameservers.
8. Inside the new zone, add:
   - **A record**, subdomain `@`, *Resolve to a Lightsail resource* → container service `flowco`.
   - **CNAME**, subdomain `www`, target = service public hostname.
9. Route 53 → **Registered domains** → `go-flow.co` → **Add or edit name servers** → replace all four with Lightsail's four. Save.
10. Wait for NS propagation (typically 5–30 min). Verify with `dig +short NS go-flow.co` — should return Lightsail NS, not Route 53's `ns-385/-1977/-1278/-867`.
11. Verify apex resolves to the container service: `dig +short go-flow.co` should match `dig +short flowco.gss8b1ryfh8jc.us-east-1.cs.amazonlightsail.com`.

### Cutover validation
12. Hit `https://go-flow.co/` in an incognito window. Confirm Streamlit loads and Google OAuth completes end-to-end.

### Decommission Fargate
13. ECS console → `flowco-cluster` → `flowco-service` → **Update service** → desired tasks **0**. Wait for the task to drain.
14. EC2 console → Load Balancers → tick `flowco-alb-1197595225` → **Actions → Delete load balancer**. (This is where most of the residual cost lived.)
15. Optional cleanup (free, but tidies things): orphaned target group, ALB security group.

### What to leave in place (rollback path)
- `.github/workflows/deploy.yml` and `.github/ecs/flowco-task.json` — untouched, but the workflow's `release: published` trigger has been **commented out** so releases no longer auto-deploy to Fargate. Only `workflow_dispatch` remains, for manual rollback runs.
- ECR repo `flowco-app` — still pushed to by the Fargate workflow when manually run. (The Lightsail workflow does not use ECR.)
- ECS task definition `flowco-task` — leave.
- ECS service `flowco-service` at desired count 0 — leave.
- CloudWatch log group `/ecs/flowco-task` — leave.
- IAM `ecsTaskExecutionRole` — leave.
- ACM cert for the ALB — leave (free).
- Route 53 hosted zone `go-flow.co` (now non-authoritative) — leave; useful for fast rollback.

### Post-cutover cleanup (2026-05-04)
Performed once the Lightsail switch was verified end-to-end:

- **Deleted Fargate ALB** `flowco-alb-1197595225` (EC2 → Load Balancers → Delete). Biggest recurring saving (~$16–22/mo).
- **Capped CloudWatch log retention** on `/ecs/flowco-task` at 30 days:
  ```sh
  aws logs put-retention-policy \
    --log-group-name /ecs/flowco-task \
    --retention-in-days 30 \
    --region us-east-1
  ```
- **Added ECR lifecycle policy** on `flowco-app` to keep only the last 10 images:
  ```sh
  aws ecr put-lifecycle-policy \
    --repository-name flowco-app \
    --region us-east-1 \
    --lifecycle-policy-text '{
      "rules": [{
        "rulePriority": 1,
        "description": "Keep last 10 images",
        "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
        "action": {"type": "expire"}
      }]
    }'
  ```
- **Disabled the `release: published` trigger** in `.github/workflows/deploy.yml` so GitHub releases only deploy to Lightsail. The Fargate workflow remains available manually via `workflow_dispatch` for rollback.

---

## 4. Migration: Lightsail → Fargate (rollback)

If you need to switch back. Assumes nothing in section 3's "What to leave in place" was deleted.

### Bring Fargate back online
1. **Restore the ALB** if it was deleted: EC2 → Load Balancers → Create ALB. Settings:
   - Internet-facing, IPv4 + IPv6 (dualstack)
   - Listeners: HTTP:80 (redirect to 443) + HTTPS:443
   - Cert: existing ACM cert for `go-flow.co`
   - Target group: new target group `flowco-tg`, type **IP**, protocol HTTP:80, VPC default, health check path `/`
   - Security group: allow 80/443 from anywhere
2. **Wire the target group to ECS** — ECS console → `flowco-service` → **Update service**:
   - Load balancing → ELB → choose the new target group
   - Desired tasks: **1**
3. Wait for the task to start and pass health checks. Confirm the ALB DNS name (`dualstack.flowco-alb-...`) loads the app over HTTPS using the ALB's cert.

(If the ALB was *not* deleted, skip 1–2 — it's still wired up. Just bump desired count to 1.)

### Flip DNS back to Route 53
4. Verify the Route 53 hosted zone for `go-flow.co` still has the original A-record aliases pointing at the ALB. If the hosted zone was deleted, recreate it and add:
   - Apex A → ALIAS to ALB
   - `www` A → ALIAS to ALB
5. Route 53 → **Registered domains** → `go-flow.co` → **Add or edit name servers** → replace Lightsail's four with the Route 53 hosted zone's four.
6. Wait for NS propagation. Verify with `dig +short NS go-flow.co`.
7. Smoke test `https://go-flow.co/`.

### Wind down Lightsail
8. Lightsail console → `flowco` container service → either:
   - **Delete the service** (stops billing immediately), or
   - Set scale to 0 if you want to keep the deployment artifacts. (Scale 0 is not a Lightsail feature — to fully pause without deleting, the only option is delete-and-recreate-later. Containers don't bill while a service is being deleted.)
9. Delete the Lightsail DNS zone (only after NS has propagated back to Route 53 — premature deletion = downtime).
10. Optionally remove the Lightsail-issued cert (it's free; can leave).

### What to leave in place (forward-rollback path)
- `.github/workflows/deploy-lightsail.yml` — untouched, ready for the next Lightsail switch.
- The IAM Lightsail policy on the GitHub Actions user — leave.

---

## 5. Cold-start: Lightsail → Fargate (Fargate side fully deleted)

Use this if the rollback-friendly resources from section 3's *What to leave in place* were torn down (e.g. ECR repo, ECS cluster/service/task definition, ALB, ACM cert, Route 53 hosted zone all gone). The only assumptions are: the AWS account `925527669208` still exists, the IAM user behind GitHub Secrets `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` still has ECS + ECR + IAM-pass-role permissions, and the source repo still has `.github/workflows/deploy.yml` + `.github/ecs/flowco-task.json`.

All steps are in `us-east-1`.

### a. ECR repository
```sh
aws ecr create-repository --repository-name flowco-app --region us-east-1
```
The URI must be `925527669208.dkr.ecr.us-east-1.amazonaws.com/flowco-app` to match the workflow.

### b. IAM execution role
Recreate `ecsTaskExecutionRole` (referenced by `flowco-task.json`):
- IAM → Roles → **Create role** → AWS service → **Elastic Container Service** → **Elastic Container Service Task** → attach managed policy `AmazonECSTaskExecutionRolePolicy` → name it **exactly** `ecsTaskExecutionRole`.

### c. ECS cluster
```sh
aws ecs create-cluster --cluster-name flowco-cluster --region us-east-1
```

### d. Register the task definition
The JSON in `.github/ecs/flowco-task.json` was exported via `describe-task-definition` and contains read-only fields (`taskDefinitionArn`, `revision`, `status`, `requiresAttributes`, `compatibilities`, `registeredAt`, `registeredBy`) that `register-task-definition` rejects as input. Strip them first:
```sh
jq 'del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)' \
  .github/ecs/flowco-task.json \
  | aws ecs register-task-definition --cli-input-json file:///dev/stdin --region us-east-1
```
Or skip this step entirely — the GitHub Actions workflow's `aws-actions/amazon-ecs-render-task-definition` step registers a clean revision on every deploy. If you go that route, run the deploy workflow once *before* step j (the `create-service` call needs at least one revision to reference).

### e. Route 53 hosted zone
If `go-flow.co`'s zone was deleted:
- Route 53 → Hosted zones → **Create hosted zone** → `go-flow.co`, public.
- Note the 4 NS records the new zone gets. **Don't update the registrar yet** — DNS still belongs to Lightsail until cutover.

### f. ACM certificate
- ACM (`us-east-1`) → **Request certificate** → public → domains `go-flow.co` and `www.go-flow.co` → DNS validation.
- ACM gives you two CNAMEs. Add them to **whichever DNS zone is currently authoritative** — at this point that's the **Lightsail** zone (since Lightsail is still serving traffic). Once DNS swings back to Route 53 in step **l**, also copy them into the Route 53 zone so renewals continue to validate.
- Wait for cert status to flip to **Issued**.

### g. Security groups
EC2 → Security Groups in the default VPC:
- `flowco-alb-sg`: inbound 80 + 443 from `0.0.0.0/0`.
- `flowco-task-sg`: inbound 80 from `flowco-alb-sg` only.

### h. Target group
EC2 → Target groups → **Create target group**:
- Type **IP addresses** (Fargate awsvpc requires IP targets, not instance).
- Protocol HTTP:80, VPC default.
- Health check path `/`, success codes `200-499` (Streamlit can return 3xx on the root).
- Name `flowco-tg`. Don't register any targets yet — ECS will manage that.

### i. Application Load Balancer
EC2 → Load Balancers → **Create** → Application Load Balancer:
- Internet-facing, **IPv4 only** (the default VPC `vpc-0de12170` has no IPv6 CIDR; dualstack mode would fail without one).
- Default VPC, public subnets in ≥2 AZs.
- Security group: `flowco-alb-sg`.
- Listeners:
  - HTTP:80 → action: redirect to HTTPS:443.
  - HTTPS:443 → certificate from step **f** → forward to `flowco-tg`.
- Name it `flowco-alb` (the original suffix `-1197595225` was AWS-assigned and can't be reproduced; section 1 references will point at the new ARN).

### j. ECS service
```sh
aws ecs create-service \
  --cluster flowco-cluster \
  --service-name flowco-service \
  --task-definition flowco-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-AAA,subnet-BBB],securityGroups=[sg-flowco-task-sg],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:925527669208:targetgroup/flowco-tg/...,containerName=flowco-container,containerPort=80" \
  --region us-east-1
```
Substitute the actual subnet IDs, security group ID, and target group ARN. Wait for the task to start and become healthy in the target group.

### k. DNS records in Route 53
Inside the new hosted zone:
- Apex `go-flow.co`: A-record, **Alias → Alias to Application and Classic Load Balancer**, region `us-east-1`, pick the new ALB.
- `www`: same pattern.

### l. Cut over from Lightsail
- Route 53 → **Registered domains** → `go-flow.co` → **Add or edit name servers** → replace Lightsail's four with the new Route 53 zone's four.
- Wait for propagation (`dig +short NS go-flow.co` should return the new Route 53 NS).
- Smoke test `https://go-flow.co/` end-to-end including Google OAuth.

### m. Wind down Lightsail
- Lightsail → `flowco` container service → **Delete service**.
- Lightsail → Domains & DNS → delete the `go-flow.co` zone (only after NS has propagated to Route 53).
- Lightsail-issued cert can be deleted or left (free either way).

### Notes
- Costs jump back to ~$50/mo: Fargate task ~$30 + ALB ~$16 + data.
- The OAuth pin `google-auth-oauthlib<1.2.0` in `requirements.txt` should stay — it's deployment-target-agnostic and equally needed on Fargate.
- `redirect_uris[1]` in `secrets.toml` and the entry in Google Cloud Console OAuth client both stay `https://go-flow.co/` — no change.

---

## Appendix — quick reference

| Resource | Fargate | Lightsail |
| --- | --- | --- |
| Build & deploy | `.github/workflows/deploy.yml` | `.github/workflows/deploy-lightsail.yml` |
| Image registry | ECR (`flowco-app`) | Lightsail internal (`:flowco.app.<n>`) |
| Compute | ECS `flowco-service` on Fargate, 1 vCPU / 2 GB | Lightsail container service `flowco`, Small |
| Public entry | ALB `flowco-alb-1197595225` | Container service public endpoint |
| TLS cert | ACM cert on ALB | Lightsail-managed cert |
| DNS authority | Route 53 hosted zone | Lightsail DNS zone |
| Apex DNS pattern | A-record ALIAS to ALB | A-record alias to container service |
| Approx monthly cost | ~$50 | ~$15 |

### Useful one-liners
```sh
# Which target is go-flow.co pointing at right now?
dig +short NS go-flow.co
dig +short go-flow.co

# Lightsail container service IP (compare to apex resolution)
dig +short flowco.gss8b1ryfh8jc.us-east-1.cs.amazonlightsail.com

# Fargate ALB IP (compare to apex resolution if rolled back)
dig +short dualstack.flowco-alb-1197595225.us-east-1.elb.amazonaws.com
```
