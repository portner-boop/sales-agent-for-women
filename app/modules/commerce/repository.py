from app.shared.entities import ScopeContext
from app.shared.storage import Database, Transaction, fetchone, require_scope


class CommerceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def transition_allowed(self, tx: Transaction, scope: ScopeContext) -> bool:
        await require_scope(tx, scope)
        blocked = await fetchone(
            tx,
            "SELECT id FROM memory_items WHERE scope_id=? AND epoch=? AND subject_id='user_self' AND validity_status='current' AND kind='boundary' AND ((predicate='offer_refusal' AND value_json='true') OR (predicate IN ('contact_permission','followup_permission') AND value_json='false')) LIMIT 1",
            (scope.scope_id, scope.epoch),
        )
        return blocked is None
