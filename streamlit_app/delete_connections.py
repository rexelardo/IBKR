from snaptrade_client import SnapTrade

CLIENT_ID = "ASK-MAIN-STREET-TEST-OGWKD"
CONSUMER_KEY = "YOUR_CONSUMER_KEY"

USER_ID = "your-user-id"
USER_SECRET = "your-user-secret"

snaptrade = SnapTrade(
    consumer_key=CONSUMER_KEY,
    client_id=CLIENT_ID,
)

# 1) List connections / authorizations
auths_resp = snaptrade.connections.list_connections(
)
auths = auths_resp.body or []

print(f"Found {len(auths)} connection(s).")

# 2) Delete each one
for a in auths:
    authorization_id = a.get("id") or a.get("authorizationId")
    name = a.get("brokerageAuthorizations", a.get("institutionName", "")) or a.get("name", "")
    print("Deleting:", authorization_id, name)

    snaptrade.connections.delete_connection(
        path_params={"authorizationId": authorization_id},
        query_params={"userId": USER_ID, "userSecret": USER_SECRET},
    )

print("Done.")