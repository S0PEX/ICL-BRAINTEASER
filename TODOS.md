# Todos

- Test smaller models when time:
  - Llama 3.2 - 1B
  - R1 - 1.5b, 7b, 8b
  - Qwen2.5 - 0.5b, 1.5b, 7b
  - Gemma2 - 2B

```
gcloud compute instances create instance-20250214-061309 \
    --project=healthy-booth-448717-s1 \
    --zone=us-central1-f \
    --machine-type=n2d-custom-8-43008 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=318654533939-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20250214-061309,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-550-v20250129,mode=rw,size=250,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```
