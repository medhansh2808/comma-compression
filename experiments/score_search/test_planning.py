"""CPU checks for score accounting and edit selection; no model dependencies."""
import unittest
import tempfile
from pathlib import Path
import shared
from token_rdo import choose_edits, independent_change, validate_proposal_state


class ScorePlanning(unittest.TestCase):
    def test_resume_validates_exact_frame_boundary(self):
        state=dict(algorithm='batched_exact_v2',archive_sha256='abc',
                   candidates_per_frame=512,shortlist_per_frame=12,
                   completed_frames=10,candidates=[dict(frame=9)])
        self.assertEqual(validate_proposal_state(state,'abc',512,12)[0],10)
        with self.assertRaises(ValueError):
            validate_proposal_state(state,'changed',512,12)
        with self.assertRaises(ValueError):
            validate_proposal_state(state,'abc',512,8)
        state['candidates'].append(dict(frame=10))
        with self.assertRaises(ValueError):
            validate_proposal_state(state,'abc',512,12)

    def test_cached_evaluation_reuses_completed_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            archive=folder/'archive.zip'
            archive.write_bytes(b'cached archive fixture')
            raw=folder/'frames.raw'
            raw.write_bytes(b'leftover temporary raw')
            value=dict(archive_sha256=shared.sha(archive),token_sha256='tokens',score=.125)
            shared.write_json(folder/'evaluation.json',value)
            result,_=shared.evaluate({'resume':True},archive,folder,expected_tokens='tokens')
            self.assertEqual(result,value)
            self.assertFalse(raw.exists())
            archive.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                shared.evaluate({'resume':True},archive,folder,expected_tokens='tokens')

    def test_single_edit_score_delta_matches_full_objective(self):
        baseline={'seg':.0001254357,'pose':1.32737e-6}
        oldseg,newseg=.0001,.00015
        oldpose,newpose=1.1e-6,1.3e-6
        delta=independent_change(baseline,oldseg,oldpose,newseg,newpose)
        expected=shared.score(baseline['seg']+(newseg-oldseg)/600,
            baseline['pose']+(newpose-oldpose)/600,163872)-shared.score(
            baseline['seg'],baseline['pose'],163872)
        self.assertAlmostEqual(delta,expected,places=14)

    def test_rate_scale_matches_archive_score(self):
        a=shared.score(.0001,1e-6,163872)
        b=shared.score(.0001,1e-6,163872-1000)
        self.assertAlmostEqual(a-b,8000*shared.RATE_PER_BIT,places=14)

    def test_selection_orders_by_score_and_separates_interactions(self):
        rows=[dict(frame=0,position=0,estimated_score_change=-1.),
              dict(frame=0,position=1,estimated_score_change=-.9),
              dict(frame=1,position=0,estimated_score_change=-.8),
              dict(frame=0,position=20,estimated_score_change=-.7),
              dict(frame=2,position=0,estimated_score_change=.1)]
        chosen=choose_edits(rows,4)
        self.assertEqual([(r['frame'],r['position']) for r in chosen],[(0,0),(1,0),(0,20)])
        self.assertEqual(len(choose_edits(rows,1)),1)


if __name__=='__main__':
    unittest.main()
